# Flexible Shared Memory - Technische Dokumentation

# Kapitel 1: Das Grundproblem und die Lösung

Wenn mehrere Python-Prozesse Daten austauschen müssen, stehen Entwickler vor einem klassischen Dilemma. Die `multiprocessing.Queue` ist einfach zu verwenden, aber langsam - jedes Objekt muss serialisiert (gepickelt) werden, was bei großen Datenstrukturen oder hohen Frequenzen zum Flaschenhals wird. Shared Memory hingegen ist schnell, da es direkten Speicherzugriff ohne Serialisierung ermöglicht, aber die Verwendung ist fehleranfällig und mühsam.

Python bietet seit Version 3.8 das `multiprocessing.shared_memory` Modul an, das einen rohen Speicherbereich zwischen Prozessen teilt. Der Entwickler erhält einen Byte-Buffer und kann darin lesen und schreiben. Das Problem dabei: Man muss das gesamte Memory-Layout selbst verwalten. Für jedes Feld in der Datenstruktur muss man ausrechnen, an welchem Byte-Offset es liegt, welche Größe es hat, und wie man es mit `struct.pack()` hineinschreibt. Bei einem simplen Sensorwert mit drei Float-Zahlen mag das noch überschaubar sein, aber sobald die Struktur komplexer wird - etwa mit Strings unterschiedlicher Länge, mehrdimensionalen Arrays oder gemischten Datentypen - wird der Code schnell unübersichtlich und fehleranfällig.

Hinzu kommt das Problem der Konsistenz. Wenn der Writer gerade dabei ist, mehrere Felder zu aktualisieren, könnte der Reader genau in diesem Moment lesen und inkonsistente Daten erhalten - zum Beispiel neue Temperatur aber alten Druck. Man muss also ein Protokoll implementieren, das dem Reader signalisiert, wann ein Datenblock vollständig und konsistent ist. Ohne einen Lock-Mechanismus verwendet man dafür oft Sequence Numbers: Der Writer schreibt eine Nummer vor und nach den Daten, der Reader prüft ob beide übereinstimmen. Aber auch das muss man selbst implementieren und testen.

@TODO: Die Einführung, Kapitl 1, ist zu simpel, einsichtig und führt zu Fedhlannahmen über dieses Modul. Muss volständig überarbeitet werden.

Flexible Shared Memory löst all diese Probleme durch einen fundamentalen Designansatz: Der Shared Memory Block wird selbstbeschreibend gemacht. Das bedeutet, dass nicht nur die Nutzdaten im Speicher liegen, sondern auch eine vollständige Beschreibung ihrer Struktur. Der Writer analysiert beim Erstellen automatisch die Python DataClass, berechnet das optimale Memory-Layout, und schreibt diese Informationen in einen Header am Anfang des Shared Memory Blocks. Der Reader kann dann diesen Header auslesen und die komplette Datenstruktur rekonstruieren - er braucht noch nicht einmal die ursprüngliche DataClass-Definition zu kennen.

Die Idee ist nicht neu. Dateiformate wie HDF5 oder selbstbeschreibende Netzwerkprotokolle verwenden ähnliche Ansätze. Aber im Bereich Python Shared Memory war dies bisher nicht standardmäßig verfügbar. Man musste entweder auf externe Bibliotheken wie NumPy's Shared Arrays zurückgreifen, die aber nur Arrays unterstützen, oder komplexe eigene Lösungen bauen.

# Kapitel 2: Anatomie eines selbstbeschreibenden Shared Memory Blocks

Um zu verstehen, wie das Modul arbeitet, schauen wir uns zunächst an, wie in diesem Modul ein Shared Memory Block intern aufgebaut ist. Der gesamte Block besteht aus drei logischen Bereichen: dem Header, optionalen FIFO-Metadaten, und den eigentlichen Daten-Slots.

Der Header beginnt mit einem festen Teil von 24 Bytes. Diese Bytes sind immer an der gleichen Position und haben eine feste Bedeutung, sodass jeder Reader sie sofort interpretieren kann, ohne etwas über die spezifische Datenstruktur zu wissen. Die ersten acht Bytes enthalten einen Hash-Wert - die letzten acht Bytes eines SHA256-Hashes über den Rest des Headers. Dieser Hash dient als Integritätsprüfung und ermöglicht es dem Reader, schnell zu erkennen, ob der Shared Memory Block mit der erwarteten Struktur übereinstimmt. Die nächsten vier Bytes geben die Gesamtlänge des Headers an, gefolgt von acht Bytes für die Gesamtgröße des kompletten Shared Memory Blocks. Die letzten vier Bytes des festen Teils enthalten die Anzahl der Slots - also ob es sich um einen einfachen Single-Slot-Modus handelt oder um einen FIFO-Buffer mit mehreren Slots.

Nach diesem festen Teil folgt der variable Teil des Headers, der die eigentliche Strukturbeschreibung enthält. Diese wird als Python-Dictionary serialisiert (gepickelt) und enthält für jedes Feld der DataClass sämtliche Informationen, die zum Lesen und Schreiben notwendig sind: den Feldnamen, den Datentyp, die Größe in Bytes, den Offset innerhalb eines Slots, und zusätzliche Metadaten je nach Typ. Bei einem String-Feld wird zum Beispiel die maximale Anzahl Zeichen gespeichert, bei einem Array die Form und der Element-Typ.

Warum wird dieser Teil gepickelt und nicht etwa als JSON oder in einem anderen Format gespeichert? Der Hauptgrund ist Einfachheit und Vollständigkeit. Pickle kann beliebige Python-Objekte serialisieren, einschließlich NumPy-Datentypen, und die Deserialisierung liefert wieder exakt die gleichen Objekte zurück. JSON würde zusätzliche Konvertierungen erfordern und könnte nicht alle Typen darstellen. Das Sicherheitsrisiko von Pickle - dass es beliebigen Code ausführen kann - ist hier akzeptabel, da nur der Header gepickelt ist, und dieser wird vom vertrauenswürdigen Writer-Prozess erstellt, nicht vom Reader.

Nach dem Header folgt, falls im FIFO-Modus gearbeitet wird, ein kleiner Metadaten-Bereich von 24 Bytes. Dieser enthält drei 64-Bit-Zahlen: den Write-Index, den Read-Index und die aktuelle Anzahl belegter Slots. Diese Zahlen koordinieren das Schreiben und Lesen im Ring-Buffer. Im einfachen Single-Slot-Modus existiert dieser Bereich nicht, da er nicht benötigt wird.

Dann kommen die eigentlichen Daten-Slots. Jeder Slot hat die gleiche Größe und das gleiche interne Layout. Am Anfang eines Slots steht eine 64-Bit Sequence Number - wir nennen sie sequence_begin. Direkt danach folgen die Status-Bytes, ein Byte für jedes Feld der Datenstruktur. Dann kommen nach einem Alignment-Padding die eigentlichen Feld-Daten in der Reihenfolge, wie sie im Layout-Dictionary definiert sind. Am Ende des Slots steht noch einmal eine 64-Bit Sequence Number - sequence_end.

Diese beiden Sequence Numbers sind der Kern des Lock-Free-Mechanismus und bilden zusammen ein klassisches "Two-Phase-Commit" Pattern für lock-freie Datenstrukturen.

Der Writer arbeitet in dieser Reihenfolge: Zunächst liest er die aktuelle sequence_begin, inkrementiert sie um eins, und schreibt den neuen Wert zurück. Dann schreibt er alle Feldwerte in den Slot. Erst ganz am Ende schreibt er die gleiche Nummer als sequence_end. In diesem Moment signalisiert er: "Die Daten sind vollständig und konsistent".

Der Reader hingegen muss in genau umgekehrter Reihenfolge arbeiten - und das ist der entscheidende Punkt für die Lock-Free-Korrektheit. Zuerst liest er sequence_end am Ende des Slots. Dann liest er alle Felder. Zuletzt liest er sequence_begin am Anfang des Slots. Nur wenn beide Sequence Numbers übereinstimmen, akzeptiert er die Daten als konsistent.

Warum ist diese umgekehrte Lesereihenfolge so entscheidend? Betrachten wir, was passieren kann: Der Writer schreibt sequence_end als allerletzte Operation. Wenn der Reader sequence_end zuerst liest und dann sequence_begin, kann er nur dann beide mit dem gleichen Wert sehen, wenn der Writer in der Zeit zwischen diesen beiden Leseoperationen nicht aktiv war. Hätte der Writer gerade geschrieben, würde der Reader entweder ein altes sequence_end sehen (Writer hat es noch nicht aktualisiert) oder ein neues sequence_begin (Writer hat schon mit dem nächsten Schreibvorgang begonnen). In beiden Fällen stimmen die Werte nicht überein, und der Reader verwirft die Daten.

Würde der Reader stattdessen vorwärts lesen - also sequence_begin zuerst - könnte folgendes passieren: Der Reader liest die neue sequence_begin (Writer hat sie gerade inkrementiert), dann liest er noch alte Feld-Daten (Writer schreibt sie gerade), und schließlich liest er das alte sequence_end (Writer hat es noch nicht aktualisiert). Beide Sequence Numbers würden übereinstimmen, obwohl die Daten inkonsistent sind - eine klassische Race Condition.

Dieser Mechanismus funktioniert ohne Locks, Mutexe oder andere Synchronisationsprimitiven. Der Writer und die Reader können vollständig unabhängig arbeiten. Der einzige Fall, in dem der Reader inkonsistente Daten liest, ist erkennbar und führt zu einem sauberen Retry. Es gibt keine Deadlocks, keine Priority Inversion, keine Race Conditions auf den Nutzdaten selbst.

# Kapitel 3: Was passiert beim Schreiben?

Betrachten wir nun den Ablauf aus Sicht des Writers. Der Entwickler definiert eine normale Python DataClass und übergibt sie dem SharedMemory-Konstruktor. Von diesem Moment an übernimmt das Modul die gesamte komplexe Arbeit.

Als erstes wird die DataClass analysiert. Das Modul iteriert über alle Felder und bestimmt für jedes dessen Typ. Dabei werden drei Hauptkategorien unterschieden: Skalare Werte, Strings, und NumPy-Arrays. Wichtig ist: Das Modul unterstützt ausschließlich diese festen Datentypen - keine Listen, Dictionaries, verschachtelte Objekte oder andere komplexe Python-Strukturen. Diese Beschränkung ist bewusst gewählt, weil nur Datentypen mit fester oder vorhersagbarer Speichergröße in Shared Memory sinnvoll sind.

Bei skalaren Werten werden primär NumPy-Typen verwendet: `np.float64`, `np.float32`, `np.int32`, `np.uint8`, `np.bool_` und so weiter. Der Grund ist einfach: NumPy garantiert eine feste Speichergröße für jeden Typ. Ein `np.float64` benötigt exakt acht Bytes, ein `np.int32` exakt vier Bytes - unabhängig von Plattform oder Wert. Ein Python `int` hingegen könnte theoretisch beliebig groß werden, was in Shared Memory nicht praktikabel ist.

Als Komfort-Feature mappt das Modul die drei Python Basis-Typen automatisch auf ihre NumPy-Entsprechungen: `float` wird zu `np.float64`, `int` zu `np.int64`, und `bool` zu `np.bool_`. Der Entwickler kann also eine DataClass mit Python-Typen definieren, und das Modul übersetzt diese intern. Dies macht den Code für einfache Fälle lesbarer, ohne die Vorteile der festen Speichergröße aufzugeben. Die Wahl von 64-Bit-Typen für Float und Integer ist bewusst - sie bieten den größten Wertebereich bei modernen 64-Bit-Systemen und vermeiden Overflow-Probleme bei typischen Anwendungsfällen.

Strings sind deutlich komplizierter als skalare Werte, weil ihre Länge variabel ist und moderne Software mit Unicode arbeiten muss. Das Modul verwendet eine spezielle Annotations-Syntax: `"str[32]"` bedeutet "Unicode-String mit maximal 32 Zeichen". Hier ist präzise Begrifflichkeit wichtig: Die Zahl 32 gibt die Anzahl der **Unicode-Zeichen** (genauer: Unicode Code Points) an, nicht die Anzahl Bytes im Speicher.

Was ist der Unterschied? In Python 3 ist ein `str` immer ein Unicode-String. Ein einzelnes Zeichen kann im Speicher unterschiedlich viele Bytes belegen, je nach verwendetem Zeichensatz. Ein lateinisches 'A' benötigt in UTF-8-Kodierung ein Byte, ein deutsches 'ä' zwei Bytes, ein chinesisches '你' drei Bytes, und ein Emoji wie '😀' sogar vier Bytes. Die Zeichenanzahl ist also nicht gleich der Byte-Anzahl.

Warum zählt das Modul Zeichen statt Bytes? Der Hauptgrund ist Benutzerfreundlichkeit und Vorhersagbarkeit. Wenn ein Entwickler `"str[32]"` definiert, erwartet er, dass 32 sichtbare Zeichen Platz haben - egal ob es sich um "Hello World" oder "你好世界" handelt. Würde man in Bytes zählen, könnten nur etwa 10 chinesische Zeichen hineinpassen, obwohl die Annotation "32" suggeriert. Beim Truncaten würde ein String wie "Müller" bei Byte 5 abgeschnitten zu "Müll" - das zweite Byte des 'ü' würde fehlen und die Dekodierung würde fehlschlagen.

Die Implementation trennt Zeichen-Truncation und Byte-Truncation sauber: Beim Schreiben prüft das Modul zunächst die Zeichenanzahl mit Python's `len(string)`, welches Unicode Code Points zählt. Ist der String zu lang, wird er auf Zeichen-Ebene gekürzt: `string[:32]`. Erst dann wird der (möglicherweise gekürzte) String UTF-8-encodiert. Als zusätzliche Sicherheit gibt es einen Byte-Check, der verhindert, dass pathologisch lange Encodings den reservierten Speicher überschreiten. Im Normalfall greift dieser aber nie, da der Zeichen-Cut bereits alles begrenzt hat.

Im Speicher wird für einen `"str[32]"`-String ein festes Layout reserviert: Vier Bytes für ein `uint32`-Längenfeld, gefolgt von einem Buffer für die UTF-8-Bytes. Die Größe dieses Buffers wird konservativ berechnet: Da ein Unicode-Zeichen theoretisch bis zu vier Bytes in UTF-8 belegen kann, wird `max_chars * 4` als Buffer-Größe verwendet. Ein `"str[32]"` reserviert also 4 + 128 = 132 Bytes. Dies ist bewusst großzügig dimensioniert, um Truncation zu vermeiden, bedeutet aber auch, dass rein ASCII-Texte viel ungenutzten Platz haben.

Für Anwendungsfälle mit reinen ASCII-Texten, bei denen Speichereffizienz wichtiger ist als Unicode-Unterstützung, kann alternativ ein Byte-Array verwendet werden: `"uint8[32]"` benötigt exakt 32 Bytes ohne Overhead. Der Entwickler muss dann allerdings selbst für korrekte String-Kodierung und -Dekodierung sorgen und die Grenzen beachten - das Modul behandelt solche Felder als rohe Zahlenarrays ohne String-Semantik.

Python unterscheidet grundsätzlich zwischen `str` (Unicode-Text) und `bytes` (rohe Byte-Sequenzen). Das aktuelle Modul unterstützt nur `str`. Eine Erweiterung um `bytes`-Felder wäre denkbar, etwa mit Syntax `"bytes[128]"` für einen festen 128-Byte-Buffer ohne Unicode-Interpretation. Dies würde Sinn machen für binäre Protokolle, Hashes, oder andere Nicht-Text-Daten. Aktuell müsste man dafür auf ein `"uint8[128]"`-Array ausweichen, was funktional gleich ist, aber semantisch weniger klar.

Arrays werden ebenfalls über eine String-Annotation definiert: `"float32[10,20]"` bedeutet ein zweidimensionales Array mit 10x20 Float32-Werten. Das Modul berechnet die Gesamtgröße als Produkt aller Dimensionen mal die Größe eines Elements. Im Speicher wird das Array flach abgelegt - ein zweidimensionales Array wird zu einem eindimensionalen Buffer. Beim Lesen wird es wieder in die richtige Form gebracht.

Nachdem alle Feldgrößen bekannt sind, berechnet das Modul das Layout. Es beginnt mit den Status-Bytes - ein Byte pro Feld - direkt nach der sequence_begin. Dann wird auf eine 8-Byte-Grenze aufgerundet, um Alignment-Anforderungen moderner CPUs zu erfüllen. Ab dieser Position werden die Felder nacheinander platziert, wobei jedes Feld an seiner berechneten Position liegt. Am Ende wird die Gesamtgröße wieder auf 8 Bytes aufgerundet.

Nun wird der Header gebaut. Das Layout-Dictionary mit allen Feldinformationen wird gepickelt. Dann werden die festen Header-Felder ausgefüllt: die Anzahl Slots, die Gesamtgröße. Der Hash wird über all diese Daten berechnet. Schließlich wird ein auto-generierter Name erzeugt - eine zufällige UUID verkürzt auf acht Hex-Zeichen, mit dem Präfix "shm_". Dieser Name ist wichtig, denn er ist der einzige Identifier, den der Reader später braucht, um den Shared Memory Block zu finden.

Jetzt kann das OS-Level Shared Memory angelegt werden. Python's `shared_memory.SharedMemory` wird mit `create=True` aufgerufen und erhält die berechnete Gesamtgröße. Das Betriebssystem reserviert einen Speicherbereich, der über den Namen erreichbar ist. Dieser Speicher ist ein vollständiger Bereich im virtuellen Adressraum, der in beiden Prozessen an unterschiedlichen Adressen gemappt sein kann, aber auf die gleichen physischen Seiten zeigt.

Der Header wird in die ersten Bytes geschrieben. Falls FIFO-Modus aktiv ist, werden die Metadaten initialisiert: write_index, read_index und count alle auf Null. Dann wird jeder Slot initialisiert. Die sequence_begin und sequence_end werden auf Null gesetzt, und alle Status-Bytes werden auf UNWRITTEN gesetzt. Damit ist der Shared Memory Block bereit.

Wenn der Writer nun Daten schreibt, passiert unterschiedliches je nach Modus. Im Single-Slot-Modus wird direkt in Slot 0 geschrieben. Die sequence_begin wird inkrementiert, die übergebenen Feldwerte werden geschrieben, die Status-Bytes aktualisiert, und sequence_end gesetzt. Im FIFO-Modus hingegen werden die Werte zunächst in einem internen Dictionary gesammelt. Erst beim Aufruf von `finalize()` wird der nächste freie Slot beschrieben, und die FIFO-Metadaten werden atomar aktualisiert.

Das Schreiben eines einzelnen Felds hängt vom Typ ab. Ein Skalar wird direkt als NumPy-Wert geschrieben - NumPy erlaubt es, einen Buffer als Array-View zu interpretieren und direkt hineinzuschreiben. Ein String wird UTF-8-encodiert, die Länge in Bytes wird geschrieben, gefolgt von den UTF-8-Bytes. Ist der String zu lang, wird er auf Zeichen-Ebene gekürzt, nicht auf Byte-Ebene - das verhindert abgeschnittene UTF-8-Sequenzen. Ein Array wird geflattened und in den Buffer kopiert.

Während des Schreibens wird für jedes Feld das Status-Byte aktualisiert. Das UNWRITTEN-Flag wird gelöscht, das MODIFIED-Flag gesetzt. Falls beim Schreiben eine Truncation stattfand - weil der String zu lang oder das Array die falsche Form hatte - wird das TRUNCATED-Flag gesetzt. Im FIFO-Modus wird zusätzlich das OVERFLOW-Flag gesetzt, falls der Buffer voll war und ein alter Slot überschrieben wurde.

# Kapitel 4: Reader-Seite - Reconstruction Magic

Der größte konzeptionelle Sprung des Moduls zeigt sich auf der Reader-Seite. Während in traditionellen Shared-Memory-Systemen beide Prozesse exakt die gleiche Struktur-Definition kennen müssen - oft durch gemeinsam importierte Header-Dateien oder DataClass-Module - kann ein Reader hier mit lediglich einem String arbeiten: dem Namen des Shared Memory Blocks. Alles andere rekonstruiert das Modul automatisch.

Betrachten wir den typischen Ablauf. Der Writer-Prozess hat einen Shared Memory Block erstellt und dabei einen auto-generierten Namen erhalten, etwa "shm_a3f8b2c1". Dieser Name wird über einen separaten Kommunikationskanal an den Reader übermittelt - typischerweise über eine Pipe, eine Queue, oder als Kommandozeilenargument beim Prozess-Start. Der Reader-Prozess ruft nun `SharedMemory("shm_a3f8b2c1")` auf, ohne überhaupt zu wissen, welche Felder die Datenstruktur hat oder wie groß sie ist. Von diesem Moment an arbeitet das Modul wie ein Archäologe, der aus gefundenen Artefakten die ursprüngliche Struktur rekonstruiert.

Der erste Schritt ist das Öffnen des OS-Level Shared Memory Blocks. Python's `shared_memory.SharedMemory` wird mit `create=False` und dem übergebenen Namen aufgerufen. Das Betriebssystem sucht in seinem internen Registry nach diesem Namen und mappt den zugehörigen Speicherbereich in den virtuellen Adressraum des Reader-Prozesses. Die virtuelle Adresse kann völlig anders sein als im Writer - auf einem 64-Bit-System könnten beide Prozesse den gleichen physischen Speicher an komplett unterschiedlichen Adressen sehen. Aber der Inhalt ist identisch, Byte für Byte.

Nun liest der Reader die ersten 24 Bytes - den festen Teil des Headers. Diese Bytes haben eine unveränderliche Struktur, die das Modul kennt: Bytes 0-7 enthalten den Hash-Wert, Bytes 8-11 die Header-Länge, Bytes 12-19 die Gesamt-Länge, und Bytes 20-23 die Anzahl Slots. Mit der Header-Länge weiß der Reader, wie weit er lesen muss, um den kompletten Header zu erfassen. Der Header könnte 200 Bytes lang sein, oder 500, oder 2000 - je nachdem wie viele Felder die DataClass hatte.

Der variable Teil des Headers wird nun gelesen und unpickelt. Pickle deserialisiert das Python-Dictionary, das für jedes Feld alle notwendigen Metadaten enthält: Name, Typ, Offset, Größe, und typspezifische Informationen. Das Dictionary für ein Feld könnte etwa so aussehen: `{'name': 'temperature', 'field_type': 'float64', 'is_scalar': True, 'offset': 16, 'size': 8}`. Der Reader hat nun eine vollständige Karte der Datenstruktur - er weiß, wo jedes Feld liegt, wie es zu interpretieren ist, und wie groß es ist.

Jetzt kommt der eigentliche Magic-Moment: Die dynamische Rekonstruktion der DataClass. Python's `dataclasses.make_dataclass()` Funktion erlaubt es, zur Laufzeit eine neue DataClass zu erstellen. Das Modul iteriert über die Feld-Metadaten und baut eine Liste von Feld-Definitionen auf. Für ein skalares Feld wie `temperature: np.float64` wird das Tupel `('temperature', np.float64)` erzeugt. Bei einem String-Feld wird die Annotation rekonstruiert: `('status', "str[32]")`. Bei einem Array: `('image', "float32[480,640,3]")`. Diese Liste wird an `make_dataclass()` übergeben, zusammen mit einem generierten Klassennamen wie `DynamicDataClass_shm_a3f8b2c1`.

Das Ergebnis ist eine vollwertige DataClass, die sich in jeder Hinsicht wie die ursprüngliche verhält. Sie hat die gleichen Feldnamen, die gleichen Typen, die gleiche Größe. Der Reader kann nun Instanzen dieser Klasse erstellen, auf Felder zugreifen, Type-Hints nutzen - alles funktioniert. Der einzige Unterschied: Die Klasse wurde nicht vom Entwickler geschrieben, sondern vom Modul aus den im Header gespeicherten Metadaten synthetisiert.

Warum ist dies so mächtig? Erstens entkoppelt es Writer und Reader vollständig. Der Writer kann seine DataClass ändern, neu kompilieren, und einen neuen Shared Memory Block erstellen. Solange der neue Block-Name übermittelt wird, kann der Reader damit arbeiten - ohne Neustart, ohne Recompilation, ohne manuelle Synchronisation der Struktur-Definitionen. Zweitens ermöglicht es generische Tools: Ein Monitoring-Tool könnte beliebige Shared Memory Blocks inspizieren, ohne deren Struktur vorher zu kennen. Drittens vereinfacht es Multi-Language-Szenarien: Ein C++-Programm könnte einen Shared Memory Block mit eigenem Header-Format erstellen, und ein Python-Adapter könnte diesen Header lesen und eine passende DataClass generieren.

Der Reader hat nun optional die Möglichkeit, die rekonstruierte Struktur zu validieren. Wenn beim Aufruf `expected_type=SensorData` übergeben wurde, berechnet das Modul den Hash dieser erwarteten Struktur genau so, wie es der Writer getan hätte. Es analysiert `SensorData`, erstellt das Layout-Dictionary, pickelt es, baut den kompletten Header-Inhalt nach, und hasht ihn. Dieser berechnete Hash wird mit dem gespeicherten Hash verglichen. Stimmen sie überein, ist garantiert, dass die Struktur exakt übereinstimmt - bis auf das letzte Byte, den letzten Offset. Stimmen sie nicht überein, gibt es eine klare Fehlermeldung mit beiden Hash-Werten, sodass der Entwickler sehen kann, dass hier unterschiedliche Versionen im Spiel sind.

Dieser Hash-basierte Vergleich ist deutlich robuster als etwa ein Vergleich von Klassennamen oder Feld-Listen. Ein Klassenname könnte zufällig gleich sein, auch wenn die Felder anders sind. Eine Feld-Liste könnte die gleichen Namen haben, aber in anderer Reihenfolge, was zu komplett falschen Offsets führen würde. Der SHA256-Hash über das gesamte serialisierte Layout fängt selbst subtilste Unterschiede ab - ein geänderter Datentyp, ein verschobener Offset, ein zusätzliches Feld, alles führt zu einem anderen Hash.

Ein interessanter Aspekt ist die Kompatibilität mit Python's `multiprocessing` Start-Methoden. Python bietet drei Modi: `fork` (Unix), `spawn` (alle Plattformen), und `forkserver`. Im `fork`-Modus wird der komplette Prozess-Speicher kopiert - DataClass-Definitionen, importierte Module, alles. Im `spawn`-Modus wird ein völlig neuer Python-Interpreter gestartet, der nur explizit übergebene Informationen erhält. Mit traditionellem Shared Memory müsste man im `spawn`-Modus sicherstellen, dass der Child-Prozess das gleiche Modul importiert und die gleiche DataClass definiert. Mit dem selbstbeschreibenden Header funktioniert beides identisch: Der Reader braucht nur den Namen, egal ob er per `fork` oder `spawn` gestartet wurde.

Die Rekonstruktion funktioniert sogar prozessübergreifend mit völlig unterschiedlichen Python-Umgebungen. Ein Writer mit Python 3.9 und NumPy 1.26 könnte einen Block erstellen, ein Reader mit Python 3.12 und NumPy 2.1 könnte ihn lesen - solange die Grund-Typen kompatibel sind. Der Header enthält Type-Namen als Strings wie "float64", die beide NumPy-Versionen verstehen. Das Modul übersetzt diese Strings zurück zu konkreten Typen, egal welche NumPy-Version läuft.

# Kapitel 5: Status-Tracking System

Eines der mächtigsten Features des Moduls ist das granulare Status-Tracking auf Feld-Ebene. Während viele Shared-Memory-Systeme nur globale Flags oder gar keine Statusinformationen bieten, weiß hier jedes einzelne Feld über seinen eigenen Zustand Bescheid. Ein Reader kann nicht nur den Wert eines Feldes lesen, sondern auch herausfinden: Ist dieser Wert vollständig und gültig? Wurde er seit dem letzten Lesen geändert? Wurde er beim Schreiben abgeschnitten? Ging durch FIFO-Overflow Daten verloren?

Das Status-Tracking System basiert auf einem Array von Bytes, eines pro Feld, direkt nach der sequence_begin im Slot. Jedes Byte kodiert fünf Flags als einzelne Bits. Die niedrigsten vier Bits sind definiert, die oberen vier für zukünftige Erweiterungen reserviert. Bit 0 (Wert 0x01) ist TRUNCATED, Bit 1 (0x02) ist UNWRITTEN, Bit 2 (0x04) ist MODIFIED, und Bit 3 (0x08) ist OVERFLOW. Diese kompakte Kodierung spart Speicher - bei 10 Feldern sind das nur 10 Bytes für alle Statusinformationen.

## 5.1 Das valid-Flag: Die wichtigste Eigenschaft

Das `valid`-Flag ist die zentrale Eigenschaft, die jeder Reader zuerst prüfen sollte. Es ist kein eigenständiges Bit im Status-Byte, sondern eine berechnete Eigenschaft: Ein Feld ist valid, wenn es weder TRUNCATED noch UNWRITTEN ist. Anders ausgedrückt: `valid = not (truncated or unwritten)`.

Die Bedeutung von `valid=False` ist eindeutig: **Die Daten sind nicht vertrauenswürdig.** Es gibt nur zwei Gründe, warum ein Feld invalid sein kann. Entweder wurde es noch nie geschrieben (UNWRITTEN), dann enthält es undefinierte Daten - typischerweise Nullen, aber das ist nicht garantiert. Oder es wurde beim Schreiben abgeschnitten (TRUNCATED), dann sind die Daten unvollständig und möglicherweise völlig unbrauchbar.

Ein typisches Verwendungsmuster sieht so aus: Der Reader prüft erst `valid`, bevor er den Wert verwendet. Code wie `if data.temperature.valid: process(data.temperature.value)` ist defensiv und robust. Dies fängt sowohl UNWRITTEN (noch keine Daten) als auch TRUNCATED (unvollständige Daten) ab. Für kritische Berechnungen - etwa Polynomkoeffizienten oder Transformationsmatrizen - ist ein truncated Wert fatal: Ein Array mit [1.0, 2.0, 3.0] ist ein völlig anderes mathematisches Objekt als [1.0, 2.0, 3.0, 4.0, 5.0]. Das TRUNCATED-Flag warnt davor.

Im Single-Slot-Modus bleibt die Validität über mehrere Writes hinweg erhalten: Einmal geschriebene Felder bleiben valid (solange sie nicht truncated werden), auch wenn sie in späteren `write()`-Aufrufen nicht mehr erwähnt werden. Die Werte im Speicher bleiben unverändert. Im FIFO-Modus hingegen ist die Semantik anders: Jeder Slot repräsentiert einen eigenständigen Datensatz, und Felder die beim `finalize()` nicht geschrieben wurden, werden explizit als UNWRITTEN markiert - selbst wenn im physischen Speicher noch alte Werte vom vorherigen Ring-Buffer-Durchlauf stehen.

## 5.2 Das UNWRITTEN-Flag: Felder ohne Daten

Das UNWRITTEN-Flag signalisiert: "Dieses Feld wurde noch nie mit einem sinnvollen Wert beschrieben." Wenn ein Slot neu initialisiert wird, setzt das Modul alle Status-Bytes auf UNWRITTEN (0x02). Dies ist der Ausgangszustand. Die eigentlichen Feld-Daten werden dabei nicht initialisiert - im Speicher steht das, was das Betriebssystem beim Anlegen des Shared Memory Blocks hineingeschrieben hat, typischerweise Nullen. Aber darauf sollte man sich nicht verlassen, es könnten theoretisch auch andere Werte sein.

Beim ersten Schreibvorgang passiert die Transformation. Der Writer schreibt den Feldwert - etwa `temperature=23.5` - und aktualisiert das Status-Byte. Das UNWRITTEN-Flag wird gelöscht (Bit wird auf 0 gesetzt), und das MODIFIED-Flag wird gesetzt (Bit wird auf 1 gesetzt). Das Status-Byte wechselt von 0x02 (UNWRITTEN) auf 0x04 (MODIFIED). Das Feld hat nun seinen ersten gültigen Wert erhalten.

Im Single-Slot-Modus bleibt UNWRITTEN=False dauerhaft bestehen, auch wenn das Feld in späteren `write()`-Aufrufen nicht erwähnt wird. Der Wert im Speicher bleibt erhalten. Ein Beispiel: `write(temperature=24.0)` schreibt temperature und löscht dessen UNWRITTEN-Flag. Ein nachfolgender `write(pressure=1013.0)` verändert pressure, lässt aber temperature unberührt - der Wert 24.0 bleibt im Speicher, UNWRITTEN bleibt False, nur MODIFIED wechselt auf False.

Im FIFO-Modus ist das Verhalten fundamental anders. Jeder Slot ist ein eigenständiger Datensatz. Beim `finalize()` werden alle Felder explizit markiert: Felder im Staging-Buffer bekommen UNWRITTEN=False, alle anderen Felder bekommen UNWRITTEN=True gesetzt. Dies verhindert, dass alte Werte aus vorherigen Ring-Buffer-Durchläufen fälschlicherweise als gültig interpretiert werden. Ein Reader, der einen FIFO-Eintrag liest bei dem nur `temperature` geschrieben wurde, sieht temperature als valid und pressure als unwritten - selbst wenn im physischen Speicher noch ein alter pressure-Wert von drei Slots zuvor steht.

## 5.3 Das MODIFIED-Flag: Änderungs-Erkennung

Das MODIFIED-Flag wird gesetzt wenn ein Feld geschrieben wird, und bleibt gesetzt bis es explizit zurückgesetzt wird. Im Single-Slot-Modus bedeutet dies: Ruft der Writer `write(temperature=24.0)` auf, wird das MODIFIED-Flag für temperature gesetzt. Ruft er danach `write(pressure=1013.0)` auf, bleibt das MODIFIED-Flag für temperature gesetzt - es wird nicht gelöscht, nur weil temperature nicht im zweiten write()-Aufruf enthalten war. Beide Felder haben nun MODIFIED=True. Die Werte akkumulieren sich.

Das einzige, was MODIFIED zurücksetzen kann, ist ein Reader-Aufruf. Standardmäßig löscht jeder `read()`-Aufruf die MODIFIED-Flags aller Felder - dies ist das erwartete Verhalten für den häufigsten Anwendungsfall: Single-Writer-Single-Reader mit Change-Detection. Der Reader ruft einfach `data = shm.read()` auf, sieht welche Felder modified sind, verarbeitet sie, und beim nächsten `read()` sind nur die Felder MODIFIED, die der Writer zwischenzeitlich geschrieben hat.

Ein typischer Single-Reader-Workflow: GUI ruft alle 100ms `read()` auf und aktualisiert nur die Widgets, deren Felder MODIFIED sind. Der Writer schreibt asynchron einzelne Felder, wenn sich Werte ändern - etwa `write(temperature=25.0)`. Beim nächsten Reader-Aufruf ist nur temperature.modified=True, die GUI aktualisiert nur dieses Widget. Dies spart Rechenzeit bei komplexen Visualisierungen oder wenn das Neuzeichnen aufwendig ist.

Bei mehreren Readern im Single-Slot-Modus wird das automatische Reset problematisch: Reader A ruft `read()` auf und löscht alle Flags. Reader B ruft danach `read()` auf und sieht alle Felder als "nicht modifiziert", obwohl sie es waren - Reader A hat die Information zerstört. Für Multi-Reader-Szenarien gibt es zwei Lösungen: Entweder nutzen alle Reader `read(reset_modified=False)` und verzichten auf Change-Detection, oder man definiert einen primären Reader (der mit Default-Reset arbeitet) und sekundäre Reader (die explizit `reset_modified=False` angeben). Letzteres erlaubt es, dass der Haupt-Reader Change-Detection nutzt, während Zusatz-Reader wie Logger oder Monitoring-Tools die Flags nicht beeinflussen.

Im FIFO-Modus funktioniert MODIFIED grundlegend anders, da jeder Slot einen eigenständigen Datensatz repräsentiert. Beim `finalize()` werden die MODIFIED-Flags bewusst gesetzt oder gelöscht, um anzuzeigen welche Felder in diesem spezifischen FIFO-Eintrag geschrieben wurden. Felder, die im `write()`-Staging-Buffer enthalten sind, bekommen MODIFIED=True. Felder, die nicht enthalten sind, bekommen MODIFIED=False - auch wenn sie in einem früheren FIFO-Eintrag geschrieben wurden. Dies ist ein wichtiger Unterschied zum Single-Slot-Modus, wo MODIFIED akkumulativ bleibt.

Der praktische Nutzen im FIFO: Ein Reader kann beim Lesen eines FIFO-Eintrags sofort erkennen, welche Felder der Writer beim Erstellen dieses Eintrags explizit gesetzt hat. Hat der Writer beispielsweise `write(temperature=25.0); finalize()` aufgerufen, ist im resultierenden Slot nur temperature.MODIFIED=True, während pressure.MODIFIED=False ist - auch wenn pressure noch einen alten Wert aus einem früheren Slot enthält. Dies hilft beim Debugging ("welche Felder hat der Writer wirklich aktualisieren wollen?") und bei partiellen Updates ("übernehme nur die Felder, die der Writer explizit gesetzt hat").

Ein interessanter Spezialfall im FIFO-Modus: `finalize()` kann auch aufgerufen werden, ohne dass vorher `write()` aufgerufen wurde. Der resultierende FIFO-Eintrag hat dann alle Felder mit MODIFIED=False und UNWRITTEN=True (falls sie noch nie geschrieben wurden) bzw. UNWRITTEN=False mit alten Werten (falls aus früherem Ring-Buffer-Durchlauf noch Daten vorhanden sind). Dies kann in Anwendungen genutzt werden, wo der Reader wissen muss, dass ein Verarbeitungsschritt stattgefunden hat - etwa als Heartbeat- oder Keep-Alive-Signal - auch wenn sich keine Daten geändert haben. Der Reader sieht dann einen neuen FIFO-Eintrag mit allen Feldern unmodified und kann daraus ableiten: "Der Writer lebt noch, auch wenn er diesmal nichts zu melden hatte."

Das automatische Reset von MODIFIED beim Reader-Aufruf ist im FIFO-Modus deaktiviert. Der `reset_modified`-Parameter wird im FIFO-Modus ignoriert und intern auf False gesetzt, unabhängig davon was der User übergibt. Dies ermöglicht es, denselben Reader-Code für Single-Slot und FIFO zu verwenden - `data = shm.read()` funktioniert in beiden Modi, verhält sich aber semantisch unterschiedlich bezüglich MODIFIED. Der Grund: Im FIFO liest der Reader bei jedem Aufruf einen anderen Slot aus dem Ring-Buffer. Ein Reset würde diesen spezifischen Slot modifizieren, während andere Slots im FIFO davon unbeeinflusst blieben. Das ergäbe keine sinnvolle Semantik. Stattdessen zeigt MODIFIED im FIFO konsistent pro Slot "welche Felder wurden bei diesem finalize() geschrieben", unabhängig davon ob und wann der Slot gelesen wird.

## 5.4 Das TRUNCATED-Flag: Warnung vor Datenverlust

Das TRUNCATED-Flag ist eine Warnung: "Der Writer wollte mehr schreiben, als Platz war." Entscheidend ist: Das Modul prüft die Größe **vor** dem Schreiben und schneidet die Daten ab, bevor sie in den Shared Memory geschrieben werden. Es gibt kein "unterbrochenes Schreiben" - stattdessen ist es ein Pre-Check mit anschließendem Slice.

Bei Strings läuft der Prozess so ab: Der Writer ruft `write(message="Long text...")` auf. Die `_write_string()`-Methode prüft zunächst die Zeichenanzahl mit Python's `len(string)`, welches Unicode Code Points zählt. Ist der String zu lang - etwa 50 Zeichen bei einem `"str[32]"`-Feld - wird er auf Zeichen-Ebene gekürzt: `string = string[:32]`. Erst dann wird der (bereits gekürzte) String UTF-8-encodiert. Als zusätzliche Sicherheit gibt es einen Byte-Check, der verhindert, dass pathologische UTF-8-Encodings den reservierten Speicher überschreiten, aber im Normalfall greift dieser nie. Schließlich wird der bereinigte String geschrieben und TRUNCATED=True gesetzt.

Bei Arrays ist der Prozess ähnlich: Die `_write_array()`-Methode konvertiert den Wert erst zu einem NumPy-Array mit dem korrekten dtype, dann wird das Array geflattened (mehrdimensional → eindimensional). Ist das flache Array zu lang - etwa 15 Elemente bei einem `"float32[10]"`-Feld - wird es per Python-Slice gekürzt: `flat_value = flat_value[:10]`. Dieser Slice ist eine instant Operation, es werden die ersten 10 Elemente extrahiert. Erst dann wird in den Shared Memory geschrieben. Falls die Form nicht stimmt oder gekürzt wurde, wird TRUNCATED=True gesetzt.

Die Konsequenzen von Truncation sind bei Strings oft harmlos: Ein Kommentar "Sensor overheated in sector 7" wird zu "Sensor overheated in sector " - lesbar, wenn auch unvollständig. Bei Arrays jedoch kann Truncation fatal sein: Ein Polynom mit Koeffizienten [1.0, 2.0, 3.0, 4.0, 5.0] ist ein völlig anderes mathematisches Objekt als [1.0, 2.0, 3.0]. Eine Transformationsmatrix mit fehlenden Zeilen ist unbrauchbar. Deshalb ist die Prüfung auf `value.valid` so wichtig - ein truncated Array ist per Definition nicht valid, und sollte nicht für Berechnungen verwendet werden.

Das TRUNCATED-Flag wird pro Feld gesetzt, unabhängig von anderen Feldern. Schreibt der Writer gemischte Daten mit einem zu langen String und einem passenden Array, ist nur das String-Feld truncated, das Array-Feld bleibt valid. Dies erlaubt es dem Reader, selektiv zu reagieren: Kritische Felder bei Truncation ablehnen, unkritische Felder trotzdem verwenden.

## 5.5 Das OVERFLOW-Flag: FIFO-Überlauf

Das OVERFLOW-Flag ist spezifisch für den FIFO-Modus und signalisiert ein Kapazitätsproblem. Wenn der Writer schneller schreibt als der Reader liest, füllt sich der FIFO-Buffer. Bei einem Buffer mit 10 Slots und einem Writer, der 11 Datensätze schreibt, bevor der Reader auch nur einen liest, ist der erste Datensatz verloren - der Writer überschreibt den ältesten Slot mit dem 11ten Datensatz. Das Modul setzt für alle Felder in diesem Slot das OVERFLOW-Flag.

Der Reader, wenn er diesen Datensatz liest, sieht: "Dies sind zwar gültige Daten, aber es gab zwischendurch Datenverlust." Die Daten im Slot selbst sind konsistent und korrekt, aber zwischen dem vorherigen gelesenen Datensatz und diesem gab es Datensätze, die übersprungen wurden. Je nach Anwendung kann das akzeptabel sein - bei Sensor-Logging, bei dem einzelne Samples fehlen dürfen - oder ein Alarm-Zustand bei kritischen Steuerungsdaten.

Das OVERFLOW-Flag ist unabhängig von den anderen Flags. Ein Feld kann gleichzeitig OVERFLOW=True und valid=True haben - die Daten sind vollständig und korrekt, aber es gab Datenverlust davor. Oder ein Feld kann OVERFLOW=True und truncated=True haben - doppeltes Problem: Datenverlust durch Überlauf plus unvollständige Daten durch Truncation.

Im Single-Slot-Modus wird OVERFLOW nie gesetzt, da es kein FIFO-Konzept gibt. Der Writer überschreibt einfach denselben Slot immer wieder, ohne dass dies als "Überlauf" betrachtet wird.

## 5.6 ValueWithStatus-Wrapper und Kopie-Semantik

Der User greift auf diese Flags über die `ValueWithStatus`-Wrapper-Klasse zu. Jedes Feld, das von `read()` zurückgegeben wird, ist nicht direkt der Wert, sondern ein Objekt, das den Wert und seinen Status kapselt. Ein Aufruf `data = shm.read()` liefert eine DataClass-Instanz, deren Felder ValueWithStatus-Objekte sind. Der Zugriff `data.temperature` liefert ein solches Objekt, nicht direkt die Float-Zahl. Das Objekt hat Properties: `value` für den eigentlichen Wert, `valid` für die Kombination "nicht truncated und nicht unwritten", `modified` für das MODIFIED-Flag, und so weiter.

Ein entscheidender Aspekt ist die Kopie-Semantik beim Lesen: Alle Daten werden aus dem Shared Memory in lokale Variablen kopiert. Bei Skalaren ist dies trivial - ein Float wird gelesen und in eine Python-Variable kopiert. Bei Strings wird der UTF-8-Buffer gelesen, dekodiert, und als Python-String zurückgegeben. Bei Arrays ruft `_read_array()` explizit `.copy()` auf dem NumPy-Array-View auf. Das bedeutet: `flat_array = np.ndarray(..., buffer=self.shm.buf, offset=offset).copy()`. Der View zeigt zunächst direkt in den Shared Memory, aber `.copy()` erstellt eine unabhängige Kopie im Heap des Reader-Prozesses.

Warum diese Kopie notwendig ist: Der Writer könnte jederzeit die Daten im Shared Memory überschreiben. Ohne Kopie hätte der Reader einen Pointer in den Shared Memory, der sich während der Nutzung ändern kann. Selbst während der Reader eine Berechnung durchführt, könnte der Writer neue Werte schreiben. Die Daten wären inkonsistent. Die Kopie garantiert, dass der Reader mit einem stabilen Snapshot arbeitet. Der Preis dafür ist der Kopier-Overhead, der bei großen Arrays (z.B. 4K-Bildern) durchaus spürbar sein kann. Dies ist aber der fundamentale Trade-Off bei Lock-Free-Systemen ohne explizite Synchronisation.

Diese Wrapper-Klasse implementiert Magic Methods für Convenience. Ein `float(data.temperature)` ruft `__float__()` auf und liefert den Wert zurück. Arithmetik wie `data.temperature + 5` funktioniert über `__add__()`. NumPy-Integration über `__array__()` erlaubt `np.array(data.image)` für Array-Felder. Der Entwickler kann also in vielen Fällen die Wrapper transparent verwenden, aber bei Bedarf die Status-Eigenschaften abfragen.

Ein praktisches Muster sieht so aus: Der Reader prüft erst `valid`, bevor er den Wert verwendet. Dies fängt sowohl UNWRITTEN (noch keine Daten) als auch TRUNCATED (unvollständige Daten) ab. Code wie `if data.temperature.valid: process(data.temperature.value)` ist defensiv und robust. Für Performance-kritische Pfade kann man die Properties cachen: `temp = data.temperature; if temp.valid and temp.modified: update_fast(float(temp))` vermeidet mehrfaches Property-Lookup.

## 5.7 Object Pooling für Performance

Ein interessantes Detail ist das Object Pooling. Bei jedem `read()` müssen für alle Felder FieldStatus- und ValueWithStatus-Objekte erstellt werden. In einem Read-Heavy-Szenario mit hunderten Reads pro Sekunde würde das viele Millionen Allokationen bedeuten - eine Last für den Garbage Collector. Das Modul verwendet daher Objekt-Pools: Bei der Initialisierung werden für jedes Feld ein FieldStatus- und ein ValueWithStatus-Objekt erstellt und in Listen gespeichert. Beim Read werden diese Objekte wiederverwendet - ihre internen Werte werden mit `_update()` aktualisiert, aber keine neuen Objekte allokiert. Dies reduziert Allokationen um etwa 83% in typischen Workloads.

## 5.8 Grenzen des Systems

Abschließend sei erwähnt, was das Status-System nicht kann. Es bietet keine Transaktionalität über mehrere Felder hinweg. Man kann nicht sagen: "Alle Felder sind modified oder keines." Jedes Feld hat seinen eigenen Status, unabhängig von den anderen. Auch gibt es keine Historie - das MODIFIED-Flag sagt nur "wurde geschrieben und nicht resettet" (Single-Slot) bzw. "wurde bei diesem finalize() geschrieben" (FIFO), nicht wann oder wie oft. Die automatische Change-Detection via `read()` im Single-Slot-Modus funktioniert nur bei Single-Writer-Single-Reader optimal - bei mehreren Readern muss man entweder alle auf `reset_modified=False` setzen (keine Change-Detection) oder einen primären Reader definieren, der die Flags nutzt und resettet, während sekundäre Reader sie unberührt lassen. Im FIFO-Modus ist Change-Detection zwischen Reads nicht möglich, da jeder Read einen anderen Slot betrifft - MODIFIED zeigt hier nur "was war in diesem Datensatz". Für komplexere Anforderungen bräuchte man höhere Abstraktionen, etwa Versionsnummern pro Feld oder Timestamps. Das aktuelle System bietet genau das, was für die meisten IPC-Szenarien ausreicht: Feld-Validität, einfache Änderungs-Erkennung im Single-Slot-Modus, und Datensatz-Metadaten im FIFO-Modus.

# Kapitel 6: FIFO-Modus im Detail

Der FIFO-Modus transformiert den Shared Memory Block von einem einfachen Wert-Container in einen Ring-Buffer für Datensätze. Diese Transformation ist nicht nur eine Erweiterung um mehrere Slots, sondern ein fundamental anderes Kommunikationsmuster mit eigenen Regeln, Garantien und Einschränkungen. Während der Single-Slot-Modus für Szenarien gedacht ist, wo nur der aktuelle Zustand zählt - etwa die Position eines Roboterarms - erlaubt der FIFO-Modus die Pufferung von Ereignissen oder Messreihen, bei denen jeder Datensatz wichtig ist.

## 6.1 Ring-Buffer Mechanismus

Die Grundlage des FIFO-Modus ist ein klassischer Ring-Buffer, auch Circular Buffer genannt. Der Shared Memory Block enthält eine feste Anzahl Slots - etwa 10 bei `slots=10` - die im physischen Speicher hintereinander liegen. Logisch sind sie aber zu einem Ring verbunden: Nach Slot 9 kommt wieder Slot 0. Dies erlaubt es, beliebig viele Datensätze zu schreiben, ohne dass der Speicher wächst - alte Daten werden einfach überschrieben.

Die Koordination zwischen Writer und Reader erfolgt über drei 64-Bit-Zahlen im Metadaten-Bereich direkt nach dem Header: write_index, read_index, und count. Diese drei Zahlen definieren vollständig den Zustand des FIFO. Der write_index zeigt auf den nächsten freien Slot, in den der Writer schreiben wird. Er zählt monoton aufwärts, von 0 bis ins Unendliche. Der tatsächliche Slot im Ring wird über Modulo berechnet: `slot = write_index % slots`. Bei slots=10 bedeutet write_index=0 Slot 0, write_index=10 wieder Slot 0, write_index=23 Slot 3.

Der read_index funktioniert analog: Er zeigt auf den nächsten Slot, den der Reader lesen wird, und zählt ebenfalls monoton aufwärts. Auch hier wird der physische Slot über Modulo bestimmt. Die Differenz zwischen write_index und read_index gibt Aufschluss über die Situation: Ist write_index=15 und read_index=10, dann hat der Writer fünf Datensätze voraus. Der Reader muss fünf Mal lesen, um aufzuholen.

Die dritte Zahl, count, gibt die aktuelle Anzahl belegter Slots an. Man könnte denken, count sei redundant - schließlich ist count=write_index-read_index. Aber das stimmt nur im Normalfall ohne Overflow. Wenn der FIFO voll läuft, beginnt der Writer, alte Slots zu überschreiben. Dabei erhöht er sowohl write_index als auch read_index (um zu signalisieren, dass der älteste Datensatz verloren ging), aber count bleibt konstant bei slots. Die Formel count=write_index-read_index würde hier fehlschlagen. Deshalb wird count explizit gespeichert und gepflegt.

Betrachten wir ein konkretes Beispiel mit slots=3. Initial sind alle Indizes 0: write_index=0, read_index=0, count=0. Der Writer ruft finalize() auf, schreibt in Slot 0, und setzt write_index=1, count=1. Noch ein finalize(): Slot 1 wird beschrieben, write_index=2, count=2. Und noch eins: Slot 2, write_index=3, count=3. Jetzt ist der FIFO voll. Ein weiteres finalize() würde Slot 0 überschreiben (write_index=3 → slot=3%3=0), und dabei entsteht der Overflow-Zustand: write_index wird 4, read_index wird auf 1 erhöht (der älteste Datensatz in Slot 0 ist jetzt ungültig), count bleibt 3.

Der Reader merkt von diesem Overflow nichts, solange er rechtzeitig liest. Liest er bevor der Overflow passiert, bekommt er die Datensätze in der richtigen Reihenfolge. Liest er nach dem Overflow, überspringt er automatisch die verlorenen Datensätze - das Modul hat read_index ja bereits erhöht. Das OVERFLOW-Flag im übersprungenen Slot signalisiert zwar "hier ging Daten verloren", aber der Reader sieht diesen Slot gar nicht, da read_index schon weiter ist. Das Flag ist eher für Debugging relevant, falls man manuell Slots inspiziert.

## 6.2 Overflow-Handling im Detail

Overflow ist der kritische Zustand im FIFO-Modus: Der Writer ist schneller als der Reader, der Buffer läuft voll, und Daten gehen verloren. Das Modul kann Overflow nicht verhindern - das wäre nur mit Locks oder Backpressure möglich, was das Lock-Free-Prinzip verletzen würde. Stattdessen macht das Modul Overflow erkennbar und dokumentiert, und überlässt es der Anwendung, darauf zu reagieren.

Der Overflow-Check findet in `_write_to_slot()` statt, unmittelbar bevor der Slot beschrieben wird. Die Logik ist einfach: `if count >= slots: overflow=True`. In diesem Moment weiß das Modul: "Ich überschreibe jetzt einen alten, noch nicht gelesenen Slot." Das OVERFLOW-Flag wird für alle Felder dieses Slots gesetzt. Nach dem Schreiben werden die Metadaten aktualisiert: write_index wird inkrementiert (wie immer), read_index wird ebenfalls inkrementiert (um den verlorenen Slot zu überspringen), und count bleibt konstant (da ein alter Slot ersetzt wurde, nicht hinzugefügt).

Aus Sicht des Readers passiert folgendes: Er ruft `read()` auf, erwartet den nächsten Datensatz, und bekommt ihn - aber mit OVERFLOW=True für alle Felder. Dies signalisiert: "Zwischen dem letzten Datensatz, den du gelesen hast, und diesem hier gab es mindestens einen Datensatz, den du nie gesehen hast." Die Daten des aktuellen Datensatzes sind vollständig korrekt und konsistent, aber die Sequenz ist unterbrochen.

Die Reaktion auf Overflow hängt stark von der Anwendung ab. Bei Sensor-Logging, wo man eine Messreihe aufzeichnet, ist gelegentlicher Overflow vielleicht akzeptabel - man hat halt einzelne Samples verloren, die Gesamttendenz ist noch erkennbar. Bei einer Steuerungsanwendung, wo jeder Datensatz ein Kommando oder ein kritisches Event ist, wäre Overflow fatal. Hier müsste die Anwendung bei detektiertem Overflow einen Alarm auslösen oder in einen Safe-State gehen.

Eine defensive Strategie ist, den FIFO-Buffer groß genug zu dimensionieren, dass Overflow unter normalen Bedingungen nie passiert. Aber wie groß ist "groß genug"? Das führt zu Sizing-Überlegungen, die wir später betrachten. Eine andere Strategie ist Monitoring: Der Reader kann zählen, wie oft er OVERFLOW sieht, und Statistiken führen. Tritt Overflow häufig auf, ist der Buffer zu klein, oder der Reader zu langsam, oder der Writer zu schnell - irgendwo muss man optimieren.

Ein subtiler Punkt: Das OVERFLOW-Flag wird im Slot gesetzt, der überschrieben wird. Der Reader, wenn er diesen Slot liest, sieht das Flag. Aber was ist mit den Datensätzen, die übersprungen wurden? Die haben OVERFLOW nicht gesetzt, weil sie zum Zeitpunkt ihres Schreibens noch kein Overflow verursachten. Sie sind einfach verloren, ohne Spur. Dies ist eine Konsequenz der Lock-Free-Architektur: Der Writer kann nicht zurückgehen und alte Slots markieren, das würde Race Conditions erzeugen.

## 6.3 latest=True Mechanismus

Der `latest`-Parameter bei `read()` erlaubt es dem Reader, direkt zum neuesten Datensatz zu springen, ohne alle zwischenliegenden zu lesen. Dies ist sinnvoll für Echtzeit-Anwendungen, wo nur der aktuelle Zustand zählt, nicht die Historie. Ein Display, das Sensor-Werte visualisiert, will vielleicht nur die neueste Messung zeigen - alte Werte sind irrelevant. Ohne `latest=True` müsste der Reader alle aufgelaufenen Datensätze durchlesen, nur um zum neuesten zu kommen. Mit `latest=True` kann er direkt dorthin springen.

Die Implementation ist erstaunlich einfach: Beim Aufruf von `_read_fifo()` wird zunächst die normale FIFO-Logik ausgeführt - write_index, read_index, count werden gelesen. Dann kommt der Check: `if latest and count > 1`. Falls True, wird read_index auf `write_index - 1` gesetzt, und count auf 1. Damit zeigt read_index auf den neuesten Datensatz, und alle älteren werden übersprungen. Die Metadaten werden entsprechend zurückgeschrieben, sodass beim nächsten Read ohne latest wieder normal weitergelesen wird.

Ein wichtiger Punkt: Die übersprungenen Datensätze werden nicht als "gelesen" markiert oder gelöscht. Sie existieren noch im Ring-Buffer, aber read_index zeigt an ihnen vorbei. Beim nächsten Writer-Durchlauf werden sie überschrieben, als wären sie nie gelesen worden. Dies ist korrekt für den Use-Case: Wenn nur der neueste Wert zählt, sind alte Werte tatsächlich irrelevant.

Der Performance-Aspekt ist subtil. Man könnte denken, `latest=True` spart Zeit, weil nicht mehrfach gelesen wird. Tatsächlich wird aber trotzdem nur ein Slot gelesen - die Slots dazwischen werden komplett übersprungen, ohne Speicherzugriff. Der Hauptvorteil ist nicht CPU-Zeit, sondern Latenz: Der Reader bekommt sofort die neuesten Daten, ohne durch einen Stapel alter Daten zu iterieren. In einem Loop `while True: data = shm.read(latest=True); display(data)` sieht der User immer den aktuellsten Wert, auch wenn der Writer zwischenzeitlich hunderte Updates gemacht hat.

Ein möglicher Fallstrick: Kombiniert man `latest=True` mit langsamen Readern, die nur selten aufrufen, kann man praktisch alle Daten verpassen. Der Reader sieht nur Snapshots im großen Abstand, ohne die Entwicklung dazwischen. Für Logging oder Analyse ist das ungeeignet - hier muss man jeden Datensatz verarbeiten, also `latest=False` verwenden. Die Wahl des Parameters hängt also stark vom Anwendungsfall ab: Echtzeit-Display → latest=True, Daten-Logger → latest=False.

## 6.4 Warum kein Multi-Reader im FIFO?

Der FIFO-Modus unterstützt explizit nur einen Reader. Dies ist eine fundamentale Einschränkung, die aus der Architektur folgt: Es gibt nur einen read_index, der im Shared Memory gespeichert ist. Wenn Reader A `read()` aufruft, wird read_index inkrementiert. Wenn nun Reader B `read()` aufruft, sieht er den inkrementierten read_index, und liest den nächsten Slot - er überspringt den Slot, den Reader A gerade gelesen hat.

Dieses Problem ist nicht einfach zu lösen, ohne die Lock-Free-Eigenschaft aufzugeben. Man könnte versuchen, pro Reader einen eigenen read_index zu speichern, aber dann müsste das Modul wissen, wie viele Reader es gibt, und wo ihre read_index-Werte liegen. Das würde zusätzliche Koordination erfordern. Man könnte auch versuchen, read_index per Compare-And-Swap atomar zu aktualisieren, aber dann müsste jeder Reader prüfen, ob sein Read erfolgreich war, und bei Kollision retry - das ist komplex und nicht mehr Lock-Free im eigentlichen Sinne.

Die einfachste Lösung ist die aktuelle: Ein FIFO-Buffer hat einen Reader. Will man mehrere unabhängige Reader, muss man mehrere FIFO-Buffer verwenden. Der Writer schreibt dann entweder in mehrere Buffer (einmal pro Reader), oder man führt eine Broadcast-Logik ein: Der primäre Reader liest den FIFO und verteilt die Daten über einen anderen Mechanismus (Queues, sekundäre Shared Memory Blöcke) an weitere Konsumenten.

Eine alternative Architektur wäre, jeden Reader seinen eigenen Ring-Buffer im gleichen Shared Memory Block geben. Der Writer schreibt dann in alle Ringe parallel. Dies ist möglich, aber kompliziert die Speicherverwaltung erheblich. Die Slot-Anzahl müsste pro Reader konfiguriert werden, der Speicherbedarf steigt linear mit der Anzahl Reader, und die Write-Performance sinkt, da der Writer in mehrere Slots schreiben muss. Für die meisten Anwendungsfälle ist es einfacher und klarer, separate FIFO-Instanzen zu verwenden.

Ein praktisches Pattern für Multi-Reader mit FIFO: Man hat einen primären Reader-Prozess, der den FIFO kontinuierlich liest und die Daten über eine multiprocessing.Queue an mehrere Worker verteilt. Die Queue unterstützt Multi-Consumer nativ. Der FIFO-Reader ist dann der einzige, der direkt auf den Shared Memory zugreift, und die Worker bekommen ihre Daten über die Queue. Dies kombiniert die Vorteile beider Welten: Lock-Free Performance zwischen Writer und primärem Reader, und flexible Multi-Consumer-Semantik über die Queue.

## 6.5 Sizing-Empfehlungen

Die Wahl der Slot-Anzahl (`slots=N`) ist eine zentrale Design-Entscheidung, die Performance, Speicherverbrauch und Robustheit gegenüber Timing-Variationen beeinflusst. Eine zu kleine Slot-Anzahl führt zu häufigem Overflow, eine zu große verschwendet Speicher und bringt ab einem gewissen Punkt keinen Vorteil mehr.

Die grundlegende Faustregel lautet: Die Slot-Anzahl sollte etwa das 2-3-fache der maximalen Anzahl Datensätze sein, die sich zwischen zwei Reader-Aufrufen aufstauen können. Diese Zahl hängt von den Frequenzen ab: Schreibt der Writer mit 100 Hz (alle 10 ms ein Datensatz), und liest der Reader mit 50 Hz (alle 20 ms), dann entstehen im Durchschnitt 2 Datensätze zwischen zwei Reads. Bei perfekt synchronisierten Timings würden `slots=2` genügen. Aber Timings sind nie perfekt - es gibt Jitter, gelegentliche Verzögerungen durch Kontext-Switches oder Garbage Collection, und Phasenverschiebungen. Daher sollte man Puffer einplanen: `slots=6` wäre hier eine sichere Wahl (3x die nominale Differenz).

Ein anderer Ansatz ist, von der maximalen tolerierbaren Latenz auszugehen. Angenommen, der Reader darf maximal 50 ms hinterherhinken, bevor Overflow passiert. Bei einem Writer mit 100 Hz (ein Datensatz alle 10 ms) entspricht das 5 Datensätzen. Mit etwas Sicherheitsabstand wären `slots=8` oder `slots=10` angemessen. Dies gibt Raum für gelegentliche Reader-Verzögerungen, ohne sofort Daten zu verlieren.

Der Speicherverbrauch skaliert linear mit der Slot-Anzahl. Ein Slot hat die Größe des kompletten Datensatzes plus Overhead (sequence numbers, status bytes). Bei einer DataClass mit 100 Bytes Nutzdaten und `slots=100` sind das etwa 10 KB. Bei großen Datensätzen - etwa 4K-Bilder mit 8 MB pro Frame - und `slots=30` wären das bereits 240 MB. Hier muss man abwägen: Ist der Speicher verfügbar? Oder muss man die Slot-Anzahl begrenzen und Overflow in Kauf nehmen?

Ein Monitoring-Ansatz zur Laufzeit: Der Reader zählt, wie oft er OVERFLOW sieht. Falls Overflow regelmäßig auftritt - etwa bei 10% aller Reads - ist der Buffer definitiv zu klein. Man sollte die Slot-Anzahl verdoppeln und beobachten, ob Overflow seltener wird. Falls Overflow nie oder extrem selten auftritt (weniger als 0.1%), könnte man die Slot-Anzahl halbieren, um Speicher zu sparen, ohne Risiko einzugehen. Dieses Trial-and-Error ist oft der pragmatischste Weg, die optimale Größe zu finden.

Ein weiterer Faktor ist die Burst-Charakteristik. Manche Systeme haben gleichmäßige Datenraten (Sensor misst konstant alle 10 ms), andere haben Bursts (Kamera liefert 10 Frames in 100 ms, dann 900 ms Pause). Bei Burst-Workloads muss der FIFO groß genug sein, um den kompletten Burst aufzunehmen, auch wenn der durchschnittliche Durchsatz niedrig ist. Eine Kamera mit 30 FPS durchschnittlich, aber 100 FPS in kurzen Bursts, bräuchte einen FIFO dimensioniert für die 100 FPS Spitze, nicht den 30 FPS Durchschnitt.

Abschließend: Es gibt keine universelle Antwort auf "wie viele Slots?". Es hängt von Writer-Frequenz, Reader-Frequenz, Timing-Jitter, Speicherbudget, und Toleranz gegenüber Overflow ab. Als Startpunkt ist `slots=10` oft eine gute Wahl für moderate Frequenzen (10-100 Hz) und moderate Datensätze (< 1 KB). Für höhere Anforderungen muss man messen, monitoren, und iterativ anpassen.

## 6.6 finalize() als atomare Operation

Das Staging-Konzept im FIFO-Modus - `write()` gefolgt von `finalize()` - ist mehr als nur eine API-Komfort-Funktion. Es implementiert eine Form von Atomarität: Entweder alle Felder im Datensatz werden zusammen committed, oder keines. Dies ist besonders wichtig, wenn ein Datensatz aus mehreren zusammengehörigen Feldern besteht, die konsistent sein müssen.

Betrachten wir ein Beispiel: Ein Roboter-Arm hat Position (x, y, z) und Geschwindigkeit (vx, vy, vz). Diese sechs Werte gehören zusammen - sie beschreiben den Zustand zu einem Zeitpunkt. Würde man sie mit sechs separaten `write()`-Aufrufen schreiben (im hypothetischen Fall, dass write() direkt schreibt), könnte der Reader zwischendurch lesen und einen inkonsistenten Zustand sehen: Position aktualisiert, aber Geschwindigkeit noch alt. Mit dem Staging-Mechanismus ist das ausgeschlossen: Der Writer ruft `write(x=..., y=..., z=...)`, dann `write(vx=..., vy=..., vz=...)`, und dann `finalize()`. Erst bei finalize() wird alles atomar in einen FIFO-Slot geschrieben.

Die Implementation des Staging ist denkbar einfach: `write()` im FIFO-Modus schreibt nicht in den Shared Memory, sondern in ein internes Dictionary `_write_buffer`. Mehrere `write()`-Aufrufe akkumulieren dort ihre Werte. `finalize()` nimmt dann den kompletten Buffer, wählt den nächsten Slot, schreibt alle Werte mit einem Aufruf von `_write_to_slot()`, und leert den Buffer. Da `_write_to_slot()` eine einzelne Funktion ist, die mit sequence numbers arbeitet, ist das Schreiben atomar aus Sicht des Readers.

Ein subtiler Punkt: Was passiert, wenn der Writer `write(x=1)`, `write(y=2)`, dann einen Crash hat, und nie `finalize()` aufruft? Der Staging-Buffer existiert nur im Heap des Writer-Prozesses, nicht im Shared Memory. Er ist verloren. Der Reader sieht nichts - kein inkonsistenter Datensatz, kein Fehler-Flag, einfach nichts. Dies ist korrekt: Ein nicht-finalisierter Datensatz ist kein Datensatz. Der Writer hatte die Absicht, einen zu schreiben, aber hat es nicht geschafft. Der Reader soll nur komplette, finalisierte Datensätze sehen.

Diese Semantik hat Vor- und Nachteile. Vorteil: Der Reader muss sich nicht mit halb-geschriebenen Daten herumschlagen. Nachteil: Wenn der Writer regelmäßig abstürzt oder `finalize()` vergisst, verliert man Daten stillschweigend. Es gibt kein "dirty-Flag" oder Warnung. Dies ist ein Trade-Off, den Lock-Free-Systeme oft machen: Konsistenz über Vollständigkeit. Lieber einen Datensatz verlieren, als inkonsistente Daten zu liefern.

Ein weiterer Aspekt ist Performance: Das Staging erlaubt es, mehrere `write()`-Aufrufe in verschiedenen Codepfaden zu machen, ohne jedes Mal Shared Memory zu berühren. Der Writer kann `write(x=...)` in einer Funktion aufrufen, `write(y=...)` in einer anderen, und erst am Ende der Verarbeitungspipeline `finalize()`. Alle Shared Memory Writes werden gebündelt. Dies reduziert die Anzahl der Speicher-Barrieren und Cache-Line-Flushes, was auf modernen CPUs durchaus messbar ist.

Schließlich der bereits erwähnte Spezialfall: `finalize()` ohne vorheriges `write()`. Dies erstellt einen FIFO-Eintrag mit allen Feldern MODIFIED=False und UNWRITTEN=True (oder mit alten Werten, falls schon mal geschrieben). Praktisch ist dies ein Heartbeat oder Keep-Alive-Signal: Der Writer signalisiert "ich lebe noch", auch wenn er diesmal keine Daten zu melden hat. Dies kann in Monitoring-Szenarien nützlich sein, um zu unterscheiden zwischen "Writer ist tot" und "Writer ist aktiv, hat aber nichts Neues". Der Reader sieht den neuen FIFO-Eintrag und weiß: Der Writer läuft, auch wenn sich keine Werte geändert haben.

# Kapitel 7: Fork vs Spawn Kompatibilität

Python's multiprocessing Modul bietet drei verschiedene Methoden, um neue Prozesse zu starten: fork, spawn, und forkserver. Diese unterscheiden sich fundamental darin, wie der neue Prozess entsteht und welchen Zustand er vom Elternprozess erbt. Viele Python-Programme funktionieren nur mit einer dieser Methoden, oder erfordern subtile Anpassungen je nach Methode. Das Flexible Shared Memory Modul hingegen funktioniert identisch mit allen drei Methoden, ohne dass der Entwickler irgendetwas ändern muss. Dies ist keine Selbstverständlichkeit, sondern eine direkte Konsequenz des selbstbeschreibenden Header-Designs.

## 7.1 Die drei Start-Methoden

Die **fork**-Methode ist der klassische Unix-Ansatz. Wenn ein neuer Prozess gestartet wird, kopiert das Betriebssystem den kompletten Speicher des Elternprozesses - alle Variablen, alle Objekte, alle importierten Module, den kompletten Zustand. Der Child-Prozess ist praktisch ein Klon des Parents zum Zeitpunkt des fork(). Dies ist extrem schnell, da moderne Betriebssysteme Copy-on-Write verwenden: Der Speicher wird nicht wirklich kopiert, nur die Page-Tables. Erst wenn Parent oder Child in eine Speicherseite schreibt, wird sie tatsächlich dupliziert.

Der Vorteil von fork ist, dass der Child-Prozess alle Definitionen kennt, die der Parent hatte. Importierte Module? Sind schon geladen. DataClass-Definitionen? Sind bekannt. Globale Variablen? Sind vorhanden. Ein Entwickler kann eine Funktion als Child-Prozess starten, und diese Funktion kann auf praktisch alles zugreifen, was im Parent definiert war. Der Nachteil ist, dass fork auf Windows nicht verfügbar ist, und dass fork mit Threads problematisch ist - nur der Thread, der fork() aufruft, existiert im Child, alle anderen sind verschwunden, was zu subtilen Bugs führen kann.

Die **spawn**-Methode ist der plattformübergreifende Standard. Der neue Prozess wird komplett neu gestartet - ein frischer Python-Interpreter, der nur das Haupt-Modul importiert und eine spezifische Funktion aufruft. Der Child kennt anfangs nichts, was nicht explizit übergeben oder importiert wurde. Will man Daten übergeben, muss man sie serialisieren (pickeln) und über eine Pipe oder Queue schicken. Will man Objekte nutzen, muss der Child das entsprechende Modul importieren.

Der Vorteil von spawn ist Plattform-Unabhängigkeit - es funktioniert auf Unix, Windows, macOS identisch. Auch ist der Child-Prozess "sauber", ohne unerwartete Nebeneffekte von geerbtem Zustand. Der Nachteil ist der höhere Overhead: Ein neuer Interpreter muss starten, Module müssen importiert werden, das dauert deutlich länger als fork. Auch ist es umständlicher: Man muss darauf achten, dass alle notwendigen Definitionen im Child-Modul verfügbar sind, entweder durch Import oder explizite Übergabe.

Die **forkserver**-Methode ist ein Hybrid. Ein spezieller Server-Prozess wird beim ersten Bedarf gestartet (per fork vom Main-Prozess). Dieser Server wartet, und wenn ein neuer Worker-Prozess benötigt wird, forked der Server einen Child. Dieser Child erbt nur den minimalen Zustand des Servers, nicht des ursprünglichen Main-Prozesses. Dies kombiniert die Geschwindigkeit von fork (ein fork ist immer noch schneller als spawn) mit der Sauberkeit von spawn (minimaler geerbter Zustand).

Die Wahl der Start-Methode hat tiefgreifende Auswirkungen auf das Design von Multi-Prozess-Programmen. Mit fork kann man lockerer sein - Dinge "funktionieren einfach", weil der Child alles kennt. Mit spawn muss man explizit sein - jede DataClass, die der Child nutzt, muss er importieren können. Traditionelle Shared-Memory-Lösungen haben hier oft Probleme: Mit fork sieht der Child die DataClass-Definition, mit spawn nicht, und das Programm bricht.

## 7.2 Warum Flexible Shared Memory mit allen Methoden funktioniert

Das Geheimnis der Kompatibilität liegt im selbstbeschreibenden Header. Wenn der Writer einen SharedMemory-Block erstellt, schreibt er die komplette Struktur-Information in den Header. Nicht nur "es gibt drei Felder", sondern "Feld 1 heißt temperature, ist np.float64, liegt bei Offset 16, hat 8 Bytes". Diese Information ist vollständig und selbst-contained. Ein Reader, der diese Information hat, braucht die ursprüngliche DataClass-Definition nicht.

Betrachten wir den fork-Fall zuerst. Der Writer-Prozess erstellt einen SharedMemory-Block mit `shm = SharedMemory(SensorData, slots=5)`. Die DataClass `SensorData` ist im Writer definiert und bekannt. Der Writer startet einen Child-Prozess per fork und übergibt den Namen: `Process(target=reader_func, args=(shm.name,))`. Der Child-Prozess erbt die komplette Speicher-Kopie des Parents, inklusive der Definition von `SensorData`. Im Child ruft man `shm = SharedMemory(shm_name, expected_type=SensorData)` auf. Das Modul liest den Header, rekonstruiert die Struktur, und validiert sie gegen `SensorData`. Alles passt, es funktioniert.

Nun der spawn-Fall, der interessanter ist. Der Writer erstellt den Block wie zuvor, aber startet den Child per spawn. Der Child ist ein frischer Interpreter, er kennt `SensorData` nicht, es sei denn er importiert das Modul, in dem es definiert ist. Aber hier kommt der Trick: Der Child muss `SensorData` gar nicht kennen! Er ruft `shm = SharedMemory(shm_name)` auf, ohne `expected_type`. Das Modul liest den Header, rekonstruiert die Struktur, und erstellt dynamisch eine neue DataClass mit `make_dataclass()`. Diese dynamische Klasse ist funktional identisch mit der originalen `SensorData` - gleiche Feldnamen, gleiche Typen, gleiches Layout. Der Child kann damit arbeiten, als hätte er die Original-Definition.

Die einzige Information, die zwischen Parent und Child fließen muss, ist der Shared-Memory-Name - ein simpler String wie "shm_a3f8b2c1". Dieser String ist trivial zu serialisieren (es ist pures ASCII) und kann über jede Kommunikations-Methode übertragen werden: Pipes, Queues, Kommandozeilen-Argumente, sogar Environment-Variablen. Keine komplexen Objekte, keine DataClass-Definitionen, keine Imports - nur ein Name.

Dies ist der fundamentale Unterschied zu traditionellen Ansätzen. Normalerweise müssten beide Prozesse die gleiche Struktur-Definition haben, entweder durch gemeinsam importierte Header-Dateien (C-Style) oder durch gemeinsam importierte Python-Module. Mit fork klappt das implizit durch Speicher-Vererbung, mit spawn bricht es, weil der Child nicht weiß, was er importieren soll. Mit dem selbstbeschreibenden Header ist die Struktur-Definition im Shared Memory selbst eingebettet, und der Child liest sie von dort.

## 7.3 Plattform-Unterschiede und Default-Methoden

Die Standard-Start-Methode hängt von der Plattform ab. Auf Unix (Linux, macOS) ist fork der Default, auf Windows ist spawn der Default (weil Windows kein fork hat). Dies führt zu einem klassischen Problem: Code, der auf Linux entwickelt wird, funktioniert prima (fork macht vieles einfach), aber bricht auf Windows (spawn ist strenger). Entwickler, die nur auf Linux testen, merken diese Inkompatibilität oft erst spät.

Das multiprocessing Modul erlaubt es, die Start-Methode explizit zu setzen: `multiprocessing.set_start_method('spawn')`. Best Practice ist, dies am Anfang des Programms zu tun und eine einheitliche Methode zu erzwingen. Noch besser: Man testet das Programm mit beiden Methoden. Wenn es mit spawn funktioniert, funktioniert es auch mit fork (spawn ist die strengere Variante). Wenn es nur mit fork funktioniert, wird es auf Windows brechen.

Mit dem Flexible Shared Memory Modul ist dieser Konflikt irrelevant. Der Code ist identisch, egal welche Start-Methode gewählt wird. Ein Programm kann sogar verschiedene Methoden für verschiedene Prozesse verwenden - etwa fork für schnelle, kurzlebige Worker, und spawn für lange laufende Prozesse, die eine saubere Umgebung brauchen. Das Modul kümmert sich nicht darum, es funktioniert überall gleich.

Ein subtiler Vorteil: Bei spawn muss der Child-Code in einer Funktion oder einem Modul sein, das importierbar ist. Man kann nicht einfach eine Lambda oder eine lokale Funktion als Target verwenden, weil der Child sie nicht importieren kann. Dies führt zu sauberem Code-Design: Die Child-Funktionen sind klar definiert, in separaten Modulen, mit klaren Interfaces. Mit fork kann man lockerer sein, was zu schlechteren Strukturen führen kann. Indem man spawn als Default wählt und damit testet, erzwingt man besseres Design.

## 7.4 Praktische Implikationen

Für den Entwickler bedeutet die Cross-Methoden-Kompatibilität weniger Kopfschmerzen. Man muss nicht daran denken, Struktur-Definitionen zu duplizieren, Imports zu koordinieren, oder Pickling-Issues zu debuggen. Man erstellt den Shared Memory, übergibt den Namen, und es funktioniert. Dies senkt die Einstiegshürde erheblich - Multiprocessing ist schon kompliziert genug, ohne zusätzliche Plattform-Spezifika.

Ein typisches Pattern sieht so aus: Der Main-Prozess erstellt die SharedMemory-Instanz und startet mehrere Worker-Prozesse, jeweils mit dem Namen als Argument. Die Worker sind identisch geschrieben, ob sie per fork oder spawn gestartet wurden. Sie attachieren per Namen, lesen Daten, verarbeiten sie, und sind fertig. Kein Import von DataClass-Definitionen, kein Koordinieren von Modul-Pfaden, keine Plattform-Spezifika.

Ein weiterer Vorteil: Das Modul kann in heterogenen Umgebungen eingesetzt werden. Ein Process könnte Python 3.9 nutzen, ein anderer Python 3.12. Solange die Grund-Typen (float64, int32, Arrays) kompatibel sind, funktioniert die Kommunikation. Der Header enthält Type-Namen als Strings, die beide Versionen verstehen. Dies ist besonders relevant in Container-Umgebungen oder bei System-Updates, wo verschiedene Prozesse mit verschiedenen Python-Versionen laufen könnten.

Schließlich: Die Kompatibilität ist keine theoretische Eigenschaft, sondern wird aktiv getestet. Die Test-Suite des Moduls parametrisiert Tests über Start-Methoden: Jeder Test wird mit fork, spawn, und forkserver ausgeführt (wo verfügbar). Falls ein Test mit einer Methode bricht, wird der Bug sofort sichtbar. Dies garantiert, dass neue Features oder Bugfixes nicht versehentlich die Cross-Methoden-Kompatibilität brechen.

Die praktische Konsequenz für Anwendungen: Man kann beruhigt spawn als Default wählen (für Windows-Kompatibilität), wissend dass das Modul gleich funktioniert. Man kann Prozesse dynamisch starten und beenden, ohne sich Gedanken über geerbten Zustand zu machen. Man kann Code auf Linux entwickeln und auf Windows deployen, ohne böse Überraschungen. Dies ist eine der Design-Säulen des Moduls: Portabilität und Robustheit über Plattformen und Konfigurationen hinweg.

# Kapitel 8: Performance-Vergleich und Trade-offs

Die Wahl zwischen multiprocessing.Queue und Shared Memory ist nicht nur eine technische, sondern auch eine Performance-Frage. Beide Mechanismen lösen das gleiche Problem - Daten zwischen Prozessen austauschen - aber auf fundamental unterschiedliche Weise mit sehr unterschiedlichen Performance-Charakteristiken. Dieses Kapitel beleuchtet, wann welcher Ansatz schneller ist, und warum.

## 8.1 Der fundamentale Unterschied: Kopieren vs Teilen

Der Queue-Ansatz basiert auf Serialisierung. Wenn ein Prozess ein Objekt in eine Queue schreibt, wird das Objekt gepickelt (in eine Byte-Sequenz umgewandelt), diese Bytes werden über eine Pipe an den anderen Prozess geschickt, und dort werden sie wieder entpickelt (zurück in ein Objekt verwandelt). Das Objekt wird also vollständig kopiert - erst in ein Byte-Format, dann über die Pipe, dann zurück ins Objekt-Format. Bei jedem Queue-Transfer entstehen zwei komplette Kopien des Objekts.

Der Shared Memory-Ansatz basiert auf gemeinsamem Speicher. Das Objekt wird einmal in einen Speicherbereich geschrieben, der von beiden Prozessen gemappt ist. Der andere Prozess liest direkt aus diesem Bereich. Theoretisch entsteht keine Kopie - beide Prozesse sehen die gleichen Bytes. Praktisch gibt es bei unserem Modul eine Kopie beim Lesen (wie in Kapitel 5.6 erklärt), aber nur eine, nicht zwei wie bei der Queue.

Diese fundamentale Differenz führt zu unterschiedlichen Performance-Profilen. Bei kleinen Objekten - etwa ein paar Integer oder Floats - ist der Pickling-Overhead gering. Die Queue kann sehr schnell sein, und der zusätzliche Aufwand von Shared Memory (Sequence Number Checks, Status-Bytes) ist relativ größer. Bei großen Objekten - etwa NumPy-Arrays mit Megabytes - dominiert der Kopier-Aufwand. Shared Memory wird drastisch schneller, weil die Kopie wegfällt (bzw. nur einmal stattfindet).

## 8.2 Pickling-Overhead

Pickling ist erstaunlich schnell für die meisten Python-Objekte. Ein simples Dictionary mit ein paar Strings und Zahlen pickelt in Mikrosekunden. NumPy-Arrays haben sogar speziell optimiertes Pickling, das fast so schnell ist wie memcpy. Aber "fast so schnell" bedeutet immer noch langsamer, und bei großen Datenmengen wird der Unterschied signifikant.

Betrachten wir konkrete Zahlen. Ein NumPy-Array mit 1920x1080 Float32-Werten (ein Full-HD Grayscale-Bild) hat etwa 8 MB. Pickling dieses Arrays dauert auf einem modernen System etwa 5-10 ms. Entpickling ähnlich. Zusammen sind das 10-20 ms pro Transfer über eine Queue. Mit Shared Memory wird das Array in etwa 2-3 ms geschrieben (ein memcpy in den Shared Memory), und in etwa 2-3 ms gelesen (ein memcpy zurück). Total 4-6 ms. Das ist 2-3x schneller.

Bei kleineren Objekten kehrt sich das um. Ein DataClass mit drei Float-Werten (24 Bytes Nutzdaten) pickelt in etwa 5-10 Mikrosekunden. Die Queue-Overhead (Pipe-Write, Kontext-Switch, Pipe-Read) ist größer als das Pickling selbst. Shared Memory braucht ähnlich lange - Sequence Number prüfen, Status-Bytes lesen, Werte kopieren. Der Unterschied ist marginal. Hier gewinnt die Queue durch Einfachheit, nicht durch Geschwindigkeit.

Ein oft übersehener Punkt: Pickling ist CPU-gebunden, Shared Memory ist Memory-Bandwidth-gebunden. Auf einem System mit vielen CPU-Cores aber langsamen RAM kann Pickling relativ gut abschneiden, da es über mehrere Cores parallelisiert werden kann (verschiedene Queue-Transfers in verschiedenen Prozessen). Shared Memory hingegen konkurriert um die Speicher-Bandwidth, die oft ein Flaschenhals ist. Auf einem High-End-Server mit schnellem RAM dreht sich das um - die Bandwidth ist reichlich, Pickling-CPU wird zum Engpass.

## 8.3 Latenz vs Durchsatz

Ein weiterer wichtiger Unterschied ist zwischen Latenz (wie lange dauert ein einzelner Transfer?) und Durchsatz (wie viele Transfers pro Sekunde?). Queues haben typischerweise niedrigere Latenz für kleine Objekte, weil die Pipe-Kommunikation vom Betriebssystem stark optimiert ist. Ein einfaches `queue.put(42)` ist extrem schnell - oft unter 10 Mikrosekunden Round-Trip.

Shared Memory hat höhere minimale Latenz, weil mehr Checks stattfinden. Sequence Numbers lesen, vergleichen, Status-Bytes interpretieren, Werte kopieren - das alles summiert sich. Selbst bei einem einzelnen Float sind es leicht 20-30 Mikrosekunden. Für Ultra-Low-Latency-Anwendungen (unter 10 µs) ist die Queue überlegen.

Beim Durchsatz dreht sich das Bild. Queues haben eine Kapazitätsgrenze - die Pipe hat einen Buffer, der voll laufen kann. Wenn der Producer schneller schreibt als der Consumer liest, blockiert der Producer. Shared Memory (speziell im FIFO-Modus) blockiert nie - der Writer überschreibt alte Daten, setzt OVERFLOW-Flags, aber macht weiter. Der maximale Durchsatz ist praktisch unbegrenzt (nur durch Speicher-Bandwidth begrenzt).

Ein Benchmark-Szenario: Producer schreibt 10.000 Datensätze so schnell wie möglich, Consumer liest sie. Mit Queue dauert das etwa 200-300 ms (Queue blockiert, Producer wartet). Mit Shared Memory FIFO (slots=100) dauert es etwa 50-80 ms (Writer rennt durch, Reader holt auf). Bei gleichmäßigen Raten ohne Bursts sind beide ähnlich schnell, aber bei Bursts gewinnt Shared Memory klar.

## 8.4 Memory Footprint

Ein oft übersehener Aspekt ist der Speicherverbrauch. Eine Queue speichert alle eingereihten Objekte. Hat die Queue 100 Einträge mit je 1 MB, sind das 100 MB. Shared Memory im FIFO-Modus mit 100 Slots braucht ebenfalls 100 MB (plus Header-Overhead). Auf den ersten Blick gleich.

Der Unterschied liegt in der Dynamik. Die Queue kann wachsen - wenn der Consumer langsam ist und der Producer weiter schreibt, wächst die Queue unbegrenzt, bis der Speicher voll ist. Shared Memory hat feste Größe - 100 Slots bleiben 100 Slots, egal wie viel geschrieben wird. Dies macht die Memory-Planung einfacher und verhindert Out-of-Memory-Situationen.

Ein weiterer Punkt: Die Queue allokiert und deallokiert ständig Objekte. Jedes `queue.get()` erstellt ein neues Python-Objekt durch Entpickling. Das belastet den Garbage Collector. Shared Memory mit Object Pooling (wie in Kapitel 5.7 beschrieben) allokiert Wrapper-Objekte nur einmal und wiederverwendet sie. Dies reduziert GC-Druck erheblich, was in GC-sensitiven Anwendungen (etwa Echtzeit-Systeme mit strikten Latenz-Garantien) entscheidend sein kann.

## 8.5 Wann Queue, wann Shared Memory?

Die Entscheidung hängt von mehreren Faktoren ab:

**Queue ist besser wenn:**

- Objekte klein sind (< 1 KB)
- Objekte komplex sind (verschachtelte Dicts, Listen, benutzerdefinierte Klassen)
- Garantierte Delivery wichtig ist (Queue verliert keine Daten, Shared Memory kann overflowén)
- Mehrere Producer oder Consumer (Queue unterstützt das nativ)
- Keine Performance-kritische Anwendung

**Shared Memory ist besser wenn:**

- Objekte groß sind (> 10 KB, speziell NumPy-Arrays)
- Hohes Datenvolumen (> 100 MB/s)
- Latenz unkritisch, Durchsatz kritisch
- Burst-Workloads (Producer viel schneller als Consumer, temporär)
- Single Producer, Single Consumer
- GC-Druck reduzieren wichtig

Ein Hybrid-Ansatz ist oft optimal: Kleine Kontroll-Nachrichten über Queue ("neues Frame verfügbar in Slot 5"), große Daten über Shared Memory (das Frame selbst). Die Queue koordiniert, Shared Memory transportiert. Dies kombiniert die Stärken beider Welten.

## 8.6 Konkrete Szenarien

**Szenario 1: Bild-Verarbeitung Pipeline**

- Kamera liefert 1920x1080 RGB-Frames bei 60 FPS
- Pipeline: Capture → Pre-Processing → Neural Net → Display
- Datengröße: 8 MB pro Frame
- Queue: ~15-20 ms pro Transfer, 3 Transfers = 50 ms → kann 60 FPS nicht halten
- Shared Memory: ~5 ms pro Transfer, 3 Transfers = 15 ms → 60 FPS easy

**Vorteil Shared Memory:** 3x schneller, ermöglicht Real-Time Processing

**Szenario 2: Sensor Logging**

- 100 Sensoren, je 10 Float-Werte, 100 Hz
- Datengröße: 100 * 10 * 4 Bytes = 4 KB pro Sample
- Queue: ~50 µs pro Sample, 100 Hz = 5 ms pro Sekunde → kein Problem
- Shared Memory: ~30 µs pro Sample, 100 Hz = 3 ms pro Sekunde → marginal besser

**Vorteil Shared Memory:** Minimal, Queue ist einfacher zu verwenden

**Szenario 3: Datenbank-Worker Pool**

- Main-Prozess verteilt SQL-Queries an Worker-Pool
- Queries sind Strings (50-500 Bytes), Results sind Dicts (variabel)
- Queue: Perfekt geeignet, natürliches Multi-Worker-Pattern
- Shared Memory: Kompliziert (jeder Worker eigener Buffer?), kein klarer Vorteil

**Vorteil Queue:** Natürlicher Fit für das Pattern

**Szenario 4: Robot Control Loop**

- Control-Prozess berechnet Sollwerte, Motor-Prozess führt aus
- Datengröße: 50 Floats (Position, Velocity, Torque pro Joint)
- Frequenz: 1 kHz (1 ms Loop)
- Queue: ~100 µs pro Transfer, aber kann blockieren bei Bursts
- Shared Memory: ~50 µs, nie blocking, FIFO puffert Bursts

**Vorteil Shared Memory:** Deterministisches Timing, kein Blocking

## 8.7 Der Copy-Overhead bei Read

Ein wichtiger Punkt, der in Kapitel 5.6 erklärt wurde: Unser Modul kopiert beim Lesen aus dem Shared Memory. Dies ist notwendig für Lock-Free-Korrektheit, bedeutet aber, dass bei großen Arrays tatsächlich ein memcpy stattfindet. Bei einem 8 MB Array sind das etwa 2-3 ms auf modernem RAM.

Dies ist immer noch schneller als Pickling (5-10 ms), aber nicht "zero-copy" im absoluten Sinne. Ein hypothetisches Modul, das direkte Pointer in den Shared Memory zurückgibt, könnte schneller sein - aber wäre unsicher, da der Writer jederzeit überschreiben kann. Der Trade-off ist bewusst gewählt: Sicherheit und Korrektheit über absolute maximale Performance.

Für Anwendungen, wo selbst dieser Copy-Overhead zu viel ist, gibt es Alternativen: Man kann einen zusätzlichen Synchronisations-Mechanismus (Locks, Semaphoren) verwenden und dann direkt auf den Shared Memory zugreifen. Aber dann verliert man die Lock-Free-Eigenschaft. Oder man arbeitet mit doppelt-gepufferten Slots, wo der Writer immer in den anderen Slot schreibt als der Reader liest - aber das ist komplex und fehleranfällig.

Das Modul wählt bewusst den Mittelweg: Eine Kopie beim Lesen, aber nur eine (nicht zwei wie bei Queue). Lock-Free, aber nicht Zero-Copy. Für die große Mehrheit der Anwendungen ist dies der richtige Trade-off.

## 8.8 Zusammenfassung: Der Performance-Sweetspot

Shared Memory glänzt bei:

- Großen Datenstrukturen (> 10 KB)
- Hohen Frequenzen (> 100 Hz)
- Single-Producer-Single-Consumer-Patterns
- Burst-Workloads mit temporärem Stau
- Anwendungen wo GC-Druck problematisch ist

Queue ist besser bei:

- Kleinen Nachrichten (< 1 KB)
- Komplexen Objekten (beliebige Python-Objekte)
- Multi-Producer-Multi-Consumer
- Wenn Delivery-Garantien wichtig sind
- Wenn Einfachheit wichtiger ist als Performance

Die Performance-Charakteristik ist nicht "Shared Memory ist immer schneller", sondern "Shared Memory ist schneller bei großen Daten und hohem Durchsatz, Queue ist schneller bei kleinen Nachrichten und niedrigem Volumen". Die Wahl sollte vom konkreten Use-Case abhängen, nicht von Dogma. In vielen Systemen ist eine Kombination optimal: Queue für Koordination, Shared Memory für Daten-Transport.

# Kapitel 9: Lock-Free Guarantees und Limitations

Das Versprechen von Lock-Free-Algorithmen ist verlockend: Keine Deadlocks, keine Priority Inversion, kein Blocking. Aber "Lock-Free" ist kein Zauberwort, das alle Probleme löst. Es ist ein spezifischer Satz von Garantien mit spezifischen Trade-offs und Einschränkungen. Dieses Kapitel erklärt präzise, was das Modul garantiert, was es nicht garantiert, und welche praktischen Konsequenzen das hat.

## 9.1 Was Lock-Free bedeutet (und was nicht)

Die formale Definition von Lock-Free lautet: "Ein Algorithmus ist Lock-Free, wenn zu jedem Zeitpunkt mindestens ein Thread Fortschritt macht, auch wenn andere Threads pausiert sind oder abstürzen." Dies ist eine schwächere Garantie als Wait-Free (jeder Thread macht Fortschritt, immer) aber stärker als Obstruction-Free (nur wenn alleine laufend macht man Fortschritt).

Im Kontext unseres Moduls bedeutet das: Der Writer kann immer schreiben, selbst wenn der Reader gerade mitten in einem Read ist. Der Reader kann immer lesen (oder retry und dann lesen), selbst wenn der Writer gerade schreibt. Keiner blockiert den anderen. Kein Mutex, kein Lock, kein Semaphore - nur atomare Operationen auf einzelnen Speicher-Locations (die 64-Bit Sequence Numbers).

Was Lock-Free **nicht** garantiert: Fairness, begrenzte Wartezeit, Starvation-Freedom. Ein Reader könnte theoretisch unendlich oft retries machen, weil der Writer zufällig immer genau dann schreibt, wenn der Reader Sequence Numbers prüft. In der Praxis ist das extrem unwahrscheinlich bei normalen Workloads, aber theoretisch möglich. Lock-Free sagt: "Irgendjemand macht Fortschritt", nicht "du machst garantiert Fortschritt".

Ein weiterer wichtiger Punkt: Lock-Free bezieht sich auf den Algorithmus, nicht auf die Hardware. Moderne CPUs garantieren, dass Reads und Writes auf aligned 64-Bit-Werten atomar sind. Darauf baut das Modul auf. Wären die Sequence Numbers 128-Bit, müssten wir explizite Atomics verwenden (wie C++ std::atomic), und auf manchen Architekturen würde das nicht funktionieren. Die Wahl von 64-Bit ist also nicht zufällig, sondern ermöglicht Lock-Free auf allen relevanten Plattformen.

## 9.2 Die Single-Writer-Einschränkung

Die kritischste Limitation des Moduls ist: **Es gibt nur einen Writer**. Zwei oder mehr Prozesse, die gleichzeitig `write()` aufrufen, führen zu Data Races und Korruption. Dies ist keine Bug, sondern ein Design-Constraint. Multi-Writer würde entweder Locks erfordern (was Lock-Free zunichte macht) oder komplexe Compare-And-Swap-Logik für jedes Feld (was Performance kostet und Komplexität erhöht).

Warum ist Multi-Writer so schwierig? Betrachten wir den Write-Ablauf: Sequence Number erhöhen, Daten schreiben, Status-Bytes setzen, Sequence Number am Ende setzen. Wenn zwei Writer gleichzeitig diesen Ablauf starten, könnte Writer A die Sequence Begin auf 5 setzen, Writer B auf 6, dann schreibt A seine Daten, B schreibt seine Daten (überschreibt A's Daten), A schreibt Sequence End=5, B schreibt Sequence End=6. Der Reader liest Sequence Begin=6, Sequence End=6, denkt "alles konsistent", aber die Daten sind ein Mix aus A's und B's Writes.

Um Multi-Writer zu unterstützen, müssten wir für jedes Feld ein Lock oder eine Compare-And-Swap-Operation haben. Jeder Write müsste prüfen: "Ist der Slot gerade in Benutzung?" Wenn ja, warten oder einen anderen Slot nehmen. Das ist möglich, aber nicht mehr einfach Lock-Free - es ist Contention Management, was eigene Komplexität und Performance-Kosten hat.

Die praktische Konsequenz: Wenn eine Anwendung mehrere Producer hat, muss ein Koordinations-Layer davor. Entweder ein Multiplexer-Prozess, der Daten von mehreren Sourcen sammelt und als einzelner Writer in Shared Memory schreibt. Oder mehrere separate Shared Memory Blöcke, einer pro Producer. Oder eine Queue vor dem Shared Memory, wo mehrere Producer einreihen und ein Consumer-Thread daraus in Shared Memory schreibt.

## 9.3 Memory Ordering und Cache Coherency

Ein subtiles aber wichtiges Thema: Moderne CPUs führen Instruktionen nicht unbedingt in der Reihenfolge aus, wie sie im Code stehen. Compiler und CPU reordern Operationen zur Optimierung, solange die Single-Thread-Semantik erhalten bleibt. Bei Multi-Threading/Multi-Processing kann das zu Problemen führen.

Python hilft hier: Der GIL (Global Interpreter Lock) serialisiert viele Operationen und erzwingt Memory Barriers implizit. In reinem Python-Code ist Memory Ordering selten ein Problem. Aber unser Modul arbeitet mit NumPy-Arrays und direktem Speicherzugriff, wo Python den GIL freigibt. Hier könnten theoretisch Reordering-Issues auftreten.

Die Praxis zeigt: Auf x86/x64-Architekturen (Intel, AMD) ist das Memory Model stark genug, dass unser Sequence-Number-Ansatz ohne explizite Memory Barriers funktioniert. x86 garantiert, dass Stores in Program Order sichtbar werden, und dass Reads die aktuellsten Werte sehen (nach einigen Cache-Coherency-Zyklen). ARM hat ein schwächeres Memory Model, dort könnten theoretisch Probleme auftreten - aber in der Praxis sind die Delays zwischen Write und Read meist groß genug, dass Cache Coherency greift.

Wäre das Modul in C++ geschrieben, müssten wir explizit `std::memory_order_acquire` und `std::memory_order_release` für die Sequence Numbers verwenden. In Python/NumPy verlassen wir uns darauf, dass die Plattform-spezifischen Garantien ausreichen. Tests auf verschiedenen Architekturen (x86, ARM) haben bisher keine Probleme gezeigt. Aber es ist wichtig zu verstehen: Dies ist ein Bereich, wo die Garantien nicht 100% formell sind, sondern auf praktischen Tests und Plattform-Kenntnissen basieren.

## 9.4 Was garantiert ist: Consistency

Die zentrale Garantie des Moduls ist: **Wenn ein Read erfolgreich ist (Sequence Numbers stimmen überein), sind die Daten konsistent.** Der Reader sieht entweder die alten Daten komplett, oder die neuen Daten komplett, aber nie einen Mix. Dies folgt direkt aus dem Sequence-Number-Protokoll: Der Reader liest Sequence End zuerst, dann die Daten, dann Sequence Begin. Wenn beide übereinstimmen, war der Writer nicht aktiv während des Reads. Die Daten sind ein gültiger Snapshot.

Diese Garantie gilt unabhängig von Timing, CPU-Cores, Cache-Hierarchien. Selbst wenn Writer und Reader auf verschiedenen CPUs laufen, mit verschiedenen Cache-Lines, ist die Konsistenz gewährleistet (nach Cache Coherency, was bei modernen Multi-Core-CPUs automatisch funktioniert).

Was **nicht** garantiert ist: Aktualität. Der Reader könnte alte Daten sehen, wenn der Writer gerade neuere geschrieben hat, aber die Sequence Numbers noch nicht aktualisiert hat. Das ist korrekt - alte Daten sind valide, nur nicht die neuesten. In Lock-Based-Systemen hätte der Reader blockiert, bis der Writer fertig ist, und dann die neuesten Daten bekommen. Hier blockiert der Reader nicht, kriegt aber vielleicht leicht veraltete Daten. Das ist der Trade-off.

## 9.5 Was nicht garantiert ist: Progress Bounds

Lock-Free garantiert nicht, dass ein Read in endlicher Zeit erfolgreich ist. Theoretisch könnte ein Reader unendlich oft retries machen, weil der Sequence Check immer fehlschlägt. In der Praxis passiert das nicht - die Wahrscheinlichkeit, dass Writer und Reader sich exakt so timen, dass jeder Read fehlschlägt, ist astronomisch gering.

Die `read()`-Methode hat ein Timeout-Argument, das dieses theoretische Problem praktisch handhabt. `read(timeout=1.0)` versucht bis zu 1 Sekunde zu lesen, und gibt dann None zurück. Das ist kein Failure, sondern eine bewusste Entscheidung: "Nach 1 Sekunde kontinuierlicher Versuche sind wir nicht erfolgreich, etwas stimmt nicht (Writer ist tot? System überlastet?)."

In normalen Workloads schlägt ein Read praktisch nie fehl. Selbst bei sehr hoher Write-Frequenz (10 kHz) und sehr langsamem Reader ist die Wahrscheinlichkeit eines Sequence-Mismatch unter 0.1%. Der Reader retries dann einfach (in einer Mikrosekunden-Schleife) und kriegt nach ein paar Versuchen valide Daten.

Ein subtiles Problem: Auf extrem ausgelasteten Systemen mit vielen Prozessen kann es passieren, dass der Reader-Prozess lange nicht scheduled wird. Der Writer schreibt munter weiter, und wenn der Reader endlich drankommt, ist der FIFO übergelaufen. Das ist kein Lock-Free-Problem, sondern ein Scheduling-Problem. Der Reader erkennt es am OVERFLOW-Flag und kann entsprechend reagieren.

## 9.6 Limitations bei großen Datensätzen

Ein praktisches Limit: Die Größe eines Datensatzes sollte "vernünftig" sein. Ein Single-Slot mit 1 GB Daten würde funktionieren, aber der Copy-Overhead beim Read wäre massiv. Der Reader bräuchte mehrere Sekunden für ein Read, während dessen der Writer weiterschreibt und den Slot überschreibt. Die Sequence Numbers würden praktisch immer mismatchen.

Die Empfehlung: Einzelne Datensätze unter 100 MB halten. Darüber wird der Copy-Overhead zu groß, und die Wahrscheinlichkeit von Sequence-Mismatches steigt. Für wirklich große Daten (> 100 MB) sollte man andere Ansätze nutzen: Daten in Chunks aufteilen, oder Shared Memory nur als Index/Pointer verwenden, wo die eigentlichen Daten als Memory-Mapped-Files liegen.

Ein weiteres Limit: Die Anzahl Felder. Das Modul erstellt für jedes Feld Status-Bytes und Wrapper-Objekte. Bei tausenden Feldern wird das overhead. Eine DataClass mit 1000 Float-Feldern würde funktionieren, aber das Status-Byte-Array wäre 1 KB groß, die Wrapper-Objekte würden Speicher fressen, und die Iteration über alle Felder beim Read wäre langsam. Die Empfehlung: Unter 100 Felder bleiben. Für größere Strukturen sollte man verschachtelte DataClasses erwägen, oder Arrays verwenden (100 Floats als ein "float32[100]"-Feld statt 100 einzelne Felder).

## 9.7 ABA-Problem und Wrap-Around

Ein klassisches Problem bei Lock-Free-Algorithmen ist das ABA-Problem: Ein Thread liest Wert A, wird unterbrochen, ein anderer Thread ändert A zu B und zurück zu A, der erste Thread wacht auf und denkt "A ist immer noch A, nichts hat sich geändert" - aber tatsächlich hat sich etwas geändert. Dies kann bei Compare-And-Swap-basierten Algorithmen zu Korruption führen.

Unser Modul hat dieses Problem nicht, weil die Sequence Numbers monoton wachsen. Selbst wenn der Writer den gleichen Wert zweimal hintereinander schreibt (temperature=23.5, dann nochmal temperature=23.5), erhöht sich die Sequence Number. Der Reader sieht: Sequence war 10, jetzt ist sie 11, also gab es einen Write. Das ist korrekt - es gab einen Write, auch wenn der Wert gleich blieb.

Ein theoretisches Problem: Sequence Number Wrap-Around. Die Sequence Numbers sind 64-Bit, also bis 2^64. Bei einem Write pro Nanosekunde würde es 584 Jahre dauern, bis Wrap-Around. In der Praxis: kein Problem. Selbst bei 1 GHz Write-Frequenz (utopisch) würde es Jahrhunderte dauern. Und selbst wenn Wrap-Around passiert, ist das unproblematisch - die Sequence Numbers starten wieder bei 0, die Vergleiche funktionieren weiter.

## 9.8 Praktische Empfehlungen

Aus den Limitations folgen praktische Design-Regeln:

1. **Single Writer einhalten**: Nie zwei Prozesse gleichzeitig write() lassen. Falls nötig, Koordinations-Layer davor bauen.
2. **Datensatz-Größe begrenzen**: Unter 10 MB pro Datensatz bleiben für gute Performance. Darüber in Chunks aufteilen.
3. **Feld-Anzahl begrenzen**: Unter 100 Felder pro DataClass. Darüber Arrays verwenden oder verschachteln.
4. **Timeouts setzen**: Immer `timeout` bei read() angeben. Auch wenn es "nie" timeout sollte, absichern gegen pathologische Fälle.
5. **Overflow monitoren**: Im FIFO-Modus OVERFLOW-Flags tracken. Häufiges Overflow bedeutet: Buffer zu klein oder Reader zu langsam.
6. **Plattform testen**: Auf den Ziel-Plattformen testen (x86, ARM, etc.). Memory-Ordering-Issues sind selten, aber plattform-spezifisch.
7. **GC-Tuning**: In GC-sensitiven Anwendungen Python-GC-Parameter anpassen. Object-Pooling hilft, aber eliminiert GC-Druck nicht komplett.
8. **Nicht für alles verwenden**: Shared Memory ist ein Tool, kein Universalmittel. Kleine Nachrichten? Queue ist einfacher. Multi-Writer? Locks oder Queue.

## 9.9 Zusammenfassung der Guarantees

**Was garantiert ist:**

- Konsistenz: Erfolgreiche Reads liefern valide Snapshots
- Lock-Freedom: Kein Deadlock, kein Blocking (außer im Timeout)
- Single-Writer-Korrektheit: Ein Writer ist sicher
- Platform-Independence: Funktioniert auf fork/spawn, Linux/Windows

**Was nicht garantiert ist:**

- Multi-Writer-Sicherheit: Zwei Writer führen zu Korruption
- Bounded Progress: Reader könnte theoretisch unendlich retries machen
- Starvation-Freedom: Reader könnte verhungern (extrem unwahrscheinlich)
- Zero-Copy: Ein memcpy beim Read bleibt (für Sicherheit)

**Das Modul ist:**

- Lock-Free für Single-Writer-Single/Multi-Reader
- Performant für große Datensätze (> 10 KB)
- Praktisch robust für normale Workloads

**Das Modul ist nicht:**

- Ein Ersatz für Locks in allen Fällen
- Für Multi-Writer geeignet
- Für winzige Nachrichten optimal (Queue ist einfacher)
- Eine theoretisch perfekte Wait-Free-Implementation

Die Lock-Free-Garantien sind stark genug für die überwiegende Mehrheit von IPC-Szenarien, aber nicht grenzenlos. Mit Verständnis der Limitations kann man das Modul effektiv und sicher einsetzen.

# =========================================================

# EINZUPFLEGENDE DOKUMENTATIONSINHALTE

-------------------------------------------------------------------

# Unterstützte Datenstrukturen und Klassen

Ein fundamentales Prinzip von Shared Memory ist, dass der benötigte Speicherplatz im Voraus bekannt sein muss. Das Betriebssystem allokiert einen Speicherblock fester Größe, der dann zwischen Prozessen geteilt wird. Dies steht im direkten Gegensatz zu Python's üblicher dynamischer Natur, wo Objekte beliebig wachsen und schrumpfen können. Diese Spannung zwischen "feste Größe vorher" und "dynamische Objekte" bestimmt, welche Datenstrukturen das Modul unterstützen kann und welche nicht.

## Warum nicht jede Python-Klasse funktioniert

Python erlaubt es, beliebig komplexe Klassen zu definieren. Eine Klasse kann Listen enthalten, die wiederum Dictionaries enthalten, die wiederum andere Objekte enthalten, in beliebiger Verschachtelung. Die Größe solcher Strukturen ist zur Laufzeit bestimmt und kann sich jederzeit ändern. Eine Liste kann von 0 auf 1000 Elemente wachsen. Ein Dictionary kann beliebig viele Schlüssel bekommen. Ein String kann beliebig lang werden.

Shared Memory hingegen verlangt: "Sage mir jetzt, wie viel Platz du brauchst, dann bekommst du ihn. Für immer. Nicht mehr, nicht weniger." Wenn wir eine Python-Liste in Shared Memory speichern wollten, müssten wir entweder ihre maximale Größe vorher festlegen (was die Flexibilität von Listen zunichte macht) oder dynamisch Speicher allokieren (was in Shared Memory extrem komplex ist und Fragmentierung verursacht).

Das grundlegende Problem ist Indirektion. Eine Python-Liste ist nicht die Daten selbst, sondern ein Pointer auf einen Heap-Bereich, wo die eigentlichen Daten liegen. Dieser Heap-Bereich ist prozess-lokal - ein Pointer aus Prozess A zeigt ins Nichts in Prozess B. Man müsste die komplette Heap-Struktur in Shared Memory nachbilden, was die Komplexität explodieren lässt.

Ein weiteres Problem ist Garbage Collection. Python-Objekte haben Reference Counts, die bestimmen, wann sie freigegeben werden. In Shared Memory müssten wir Reference Counting über Prozessgrenzen hinweg koordinieren - ein Alptraum. Welcher Prozess ist verantwortlich fürs Freigeben? Was passiert, wenn ein Prozess abstürzt, während er eine Referenz hält?

Die Konsequenz: Wir müssen uns auf Datentypen beschränken, die diese Probleme nicht haben. Typen mit fester Größe, ohne Indirektion, ohne dynamische Allokation, ohne GC-Probleme. Das klingt restriktiv, ist aber für IPC-Szenarien meist ausreichend.

## Das DataClass-Requirement

Das Modul arbeitet ausschließlich mit Python DataClasses. Eine DataClass ist eine spezielle Form von Klasse, die mit dem `@dataclass`-Decorator markiert ist und primär als Container für Daten dient. Der Vorteil von DataClasses für unser Modul ist, dass sie explizite Type-Annotations haben. Jedes Feld hat einen deklarierten Typ, und diese Information ist zur Laufzeit verfügbar über `dataclasses.fields()`.

```python
from dataclasses import dataclass

@dataclass
class SensorData:
    temperature: float
    pressure: float
    timestamp: float
```

Diese Typ-Annotations erlauben es dem Modul, das Memory-Layout vorab zu berechnen. Es iteriert über die Felder, analysiert ihre Typen, berechnet die benötigten Bytes, und legt das Layout fest - alles bevor der erste Wert geschrieben wird. Die DataClass selbst ist nur eine Spezifikation, die zur Memory-Layout-Berechnung dient. Die tatsächlichen Daten liegen dann im Shared Memory, nicht im DataClass-Objekt.

Ein wichtiger Punkt: Das Modul erwartet, dass die Type-Annotations tatsächlich den Datentyp beschreiben, nicht nur Hints sind. In normalem Python-Code sind Type-Annotations optional und werden zur Laufzeit ignoriert. Hier sind sie essentiell. Das Modul liest `temperature: float`, interpretiert das als "dieses Feld ist ein Float", und reserviert 8 Bytes dafür. Ist die Annotation falsch oder fehlt, funktioniert das Layout nicht.

DataClasses haben noch einen weiteren Vorteil: Stabile Feld-Reihenfolge. Seit Python 3.7 ist die Reihenfolge der Felder in einer DataClass garantiert die Reihenfolge, wie sie im Code definiert sind. Dies ist wichtig für Memory-Layout-Konsistenz zwischen verschiedenen Prozessen. Wären die Felder in zufälliger Reihenfolge, könnte das Layout zwischen Writer und Reader differieren, selbst wenn beide die gleiche DataClass-Definition haben.

## Unterstützte primitive Typen

Die fundamentalen Bausteine sind NumPy-Skalare. NumPy definiert exakte Datentypen mit fester Bit-Breite: `np.float64` (8 Bytes), `np.float32` (4 Bytes), `np.int64` (8 Bytes), `np.int32` (4 Bytes), `np.int16` (2 Bytes), `np.int8` (1 Byte), und so weiter. Diese Typen haben keine Überraschungen - ein `np.int32` ist immer exakt 32 Bit, auf jeder Plattform, in jedem Kontext.

Als Komfort-Feature unterstützt das Modul auch Python's primitive Typen `float`, `int`, und `bool`, mappt sie aber intern auf NumPy-Typen. Ein Python `float` wird zu `np.float64`, ein `int` zu `np.int64`, ein `bool` zu `np.bool_`. Dies erlaubt einfachere DataClass-Definitionen für Einsteiger, ohne dass man NumPy-Typen kennen muss. Aber unter der Haube sind es immer NumPy-Typen, weil nur die garantierte Größen haben.

```python
@dataclass
class SimpleData:
    value: float        # → np.float64 (8 Bytes)
    count: int          # → np.int64 (8 Bytes)
    active: bool        # → np.bool_ (1 Byte)
```

Diese Mapping-Entscheidung wurde bewusst so getroffen: `float` zu `np.float64` (nicht `float32`) gibt maximale Präzision, `int` zu `np.int64` (nicht `int32`) gibt maximalen Wertebereich. Bei Speicher-kritischen Anwendungen kann man explizit `np.float32` oder `np.int32` verwenden, um Platz zu sparen. Das Modul zwingt niemanden zu 64-Bit, bietet es aber als komfortablen Default.

## Strings mit fester Kapazität

Strings sind die erste nicht-triviale Herausforderung. Ein Python-String kann beliebig lang sein, von 0 Zeichen bis Gigabytes. Aber im Shared Memory brauchen wir eine feste Größe. Die Lösung ist eine explizite Längen-Annotation: `"str[32]"` bedeutet "ein String mit maximal 32 Zeichen".

```python
@dataclass
class MessageData:
    status: "str[32]"
    details: "str[128]"
```

Die Zahl in den Klammern ist die maximale Anzahl Unicode-Zeichen, nicht Bytes. Dies ist wichtig für Nicht-ASCII-Texte. Ein chinesisches Zeichen zählt als ein Zeichen, auch wenn es in UTF-8 drei oder vier Bytes belegt. Im Speicher reserviert das Modul konservativ `max_chars * 4 + 4` Bytes: Vier Bytes pro Zeichen (UTF-8 worst case) plus vier Bytes für die Längen-Information.

Ein `"str[32]"`-Feld belegt also 132 Bytes (4 + 32*4). Dies ist großzügig dimensioniert - für reine ASCII-Texte würde `max_chars * 1` reichen. Aber Unicode-Unterstützung erfordert diese Großzügigkeit. Ein Emoji oder ein seltenes chinesisches Zeichen kann tatsächlich vier Bytes brauchen, und wir wollen kein Truncation aufgrund von Encoding-Problemen.

Die String-Länge muss zur Compile-Zeit bekannt sein - sie ist Teil der Type-Annotation. Man kann nicht zur Laufzeit entscheiden "dieser String soll 50 Zeichen erlauben". Dies mag restriktiv erscheinen, ist aber die einzige Möglichkeit, ein festes Memory-Layout zu gewährleisten. In der Praxis wählt man eine vernünftige Obergrenze: Kurze IDs bekommen `str[16]`, Beschreibungen `str[64]`, längere Texte `str[256]`. Passt ein Text nicht, wird er truncated, und das TRUNCATED-Flag signalisiert es.

## Arrays mit fester Form

NumPy-Arrays sind mächtige Datenstrukturen für wissenschaftliches Computing. Ein 2D-Array kann eine Matrix repräsentieren, ein 3D-Array ein RGB-Bild, ein 1D-Array eine Zeitreihe. Das Modul unterstützt Arrays beliebiger Dimensionalität, solange Form und Datentyp fest sind.

```python
@dataclass
class ImageData:
    frame: "float32[480,640,3]"     # RGB-Bild: 480 Höhe, 640 Breite, 3 Kanäle
    histogram: "uint32[256]"        # Histogramm: 256 Bins
    matrix: "float64[4,4]"          # Transformationsmatrix: 4x4
```

Die Annotation `"float32[480,640,3]"` spezifiziert drei Dinge: Element-Typ (`float32`), Form (`480,640,3`), und damit implizit die Gesamtgröße (480 * 640 * 3 * 4 Bytes = 3.6 MB). Im Shared Memory wird das Array flach abgelegt - die Mehrdimensionalität ist logisch, nicht physisch. Die 3D-Form wird zu einem 1D-Buffer mit 921600 Elementen.

Beim Schreiben konvertiert das Modul das NumPy-Array in dieses flache Format. Beim Lesen wird es zurück in die deklarierte Form gebracht. Der Entwickler arbeitet immer mit mehrdimensionalen Arrays, das Modul handhabt das Flattening und Reshaping transparent. Ein wichtiger Punkt: Die Form ist Teil des Typs. Ein Array `[480,640,3]` ist nicht kompatibel mit einem Array `[640,480,3]` - die Dimensionen müssen exakt übereinstimmen, sonst wird TRUNCATED gesetzt.

Die unterstützten Element-Typen für Arrays sind die NumPy-Numerik-Typen: `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`. Keine komplexen Zahlen (die sind selten in IPC-Szenarien), keine strukturierten Arrays (die sind zu komplex für feste Layouts), keine Object-Arrays (die enthalten Pointer).

## Was nicht unterstützt wird

Um Klarheit zu schaffen, hier eine explizite Liste dessen, was **nicht** funktioniert:

**Dynamische Kollektionen:**
- `List[int]` - Listen haben keine feste Länge
- `Dict[str, float]` - Dictionaries haben variable Größe
- `Set[int]` - Sets sind dynamisch
- `Tuple[int, ...]` - Variable-Length Tuples

Diese Typen erfordern dynamische Allokation, was Shared Memory nicht bietet. Selbst `Tuple[int, int, int]` mit fester Länge funktioniert nicht, weil das Modul keine Tuple-spezifische Logik hat. Man würde stattdessen drei separate `int`-Felder verwenden, oder ein `"int32[3]"`-Array.

**Verschachtelte Strukturen:**
- Dataclass innerhalb einer Dataclass
- Arrays von Dataclasses
- Strings innerhalb von Arrays

Das Modul unterstützt keine Verschachtelung. Jedes Feld ist ein primitiver Typ (Skalar, String, Array). Man kann nicht eine `Address`-Dataclass als Feld in einer `Person`-Dataclass haben. Die Lösung ist Flattening: Statt `address: Address` mit Feldern `street` und `city`, definiert man `address_street: "str[64]"` und `address_city: "str[32]"` direkt in der äußeren Klasse.

**Komplexe Python-Typen:**
- `datetime` - Ist intern komplex
- `Decimal` - Keine feste Größe
- `enum.Enum` - Kann funktionieren, aber nicht direkt unterstützt (man würde den Integer-Wert speichern)
- Custom Classes - Ohne `@dataclass` nicht unterstützt

Diese Typen haben entweder keine feste Größe, oder komplexe interne Strukturen, die das Modul nicht kennt. Für `datetime` würde man typischerweise ein Unix-Timestamp (Float oder Int) speichern. Für `Enum` speichert man den Integer-Wert und konvertiert beim Lesen zurück.

**Pointer-artige Strukturen:**
- Callbacks, Funktionen
- File-Handles
- Socket-Objekte
- Thread-Locks

Diese Dinge sind per Definition prozess-lokal und können nicht geteilt werden. Ein File-Handle aus Prozess A ist in Prozess B ungültig. Man kann keine Funktion über Prozessgrenzen "teilen". Diese Konzepte sind orthogonal zu Shared Memory.

## Verschachtelung und Workarounds

Das Fehlen von Verschachtelung ist eine bewusste Design-Entscheidung. Verschachtelte Strukturen würden das Memory-Layout exponentiell verkomplizieren. Ein Array von DataClasses wäre ein Array von variabel großen Objekten - wie legt man das flach ab? Ein DataClass mit einem DataClass-Feld würde rekursive Layout-Berechnung erfordern, mit allen Edge-Cases (zirkuläre Referenzen? Self-Referenzen?).

Stattdessen fordert das Modul: Flache Strukturen. Alle Felder auf einer Ebene. Dies mag simpel erscheinen, ist aber für IPC-Szenarien meist ausreichend. Sensordaten? Flache Liste von Messwerten. Roboter-State? Position, Velocity, Torque pro Joint, alles flach. Bildverarbeitung? Ein großes Array plus ein paar Metadaten-Felder.

Falls echte Verschachtelung notwendig ist, gibt es Workarounds:

**Workaround 1: Flattening**
```python
# Statt:
@dataclass
class Address:
    street: str
    city: str

@dataclass
class Person:
    name: str
    address: Address

# Verwende:
@dataclass
class Person:
    name: "str[64]"
    address_street: "str[64]"
    address_city: "str[32]"
```

**Workaround 2: Multiple Shared Memory Blocks**
Für komplexere Strukturen kann man mehrere SharedMemory-Instanzen verwenden. Ein Block für `Person`, ein anderer für `Address`, mit einem Linking-Mechanismus (z.B. IDs). Dies ist aufwendiger, aber ermöglicht beliebige Komplexität.

**Workaround 3: Serialisierung in Byte-Array**
Für wirklich dynamische Daten kann man sie in ein Byte-Array serialisieren (z.B. JSON oder Pickle) und dieses Array als `"uint8[N]"`-Feld speichern. Dies verliert die Type-Safety und Field-Level-Status, ist aber flexibel. Nur als letzter Ausweg empfohlen.

## Default-Werte und Initialisierung

DataClasses erlauben Default-Werte für Felder. Diese werden vom Modul ignoriert beim Layout-Berechnen, sind aber relevant für die Python-seitige Nutzung:

```python
@dataclass
class ConfigData:
    enabled: bool = False
    timeout: float = 1.0
    name: "str[32]" = ""
```

Der Default `= False` bedeutet: Wenn man ein `ConfigData()`-Objekt in Python erstellt, ist `enabled` initial False. Aber im Shared Memory? Der Slot wird mit UNWRITTEN initialisiert, und die eigentlichen Daten sind undefiniert (typischerweise Nullen, aber nicht garantiert). Der Default-Wert beeinflusst nicht, was im Shared Memory steht - nur was im Python-Objekt steht, wenn man es ohne Argumente erstellt.

Dies ist ein subtiler Punkt: Defaults sind reine Python-Semantik, keine Shared-Memory-Semantik. Ein Reader, der ein UNWRITTEN-Feld liest, bekommt einen undefiniert-Wert zurück, nicht den Default. Möchte man einen "Default" im Shared Memory, muss der Writer explizit schreiben: `shm.write(enabled=False, timeout=1.0)` beim ersten Mal.

## Größenbeschränkungen und Pragmatik

Theoretisch könnte man riesige Strukturen definieren - ein `"float32[10000,10000]"`-Array wäre 400 MB. Praktisch sollte man vernünftig sein. Sehr große Strukturen haben mehrere Probleme:

1. **Allokations-Zeit**: Das Betriebssystem braucht Zeit, 400 MB zu allokieren und zu nullen.
2. **Copy-Overhead**: Das Modul kopiert beim Lesen, 400 MB zu kopieren dauert Dutzende Millisekunden.
3. **Cache-Effizienz**: Riesige Strukturen passen nicht in CPU-Caches, Zugriffe sind langsam.
4. **Sequence-Check-Window**: Während der Reader 400 MB liest, könnte der Writer mehrfach überschreiben.

Die Empfehlung: Einzelne Felder unter 10 MB halten, Gesamt-Struktur unter 100 MB. Darüber sollte man überdenken, ob Shared Memory der richtige Ansatz ist, oder ob man die Daten anders strukturieren sollte (Chunking, Streaming, Memory-Mapped Files).

## Methoden in DataClasses: Nur Daten, kein Code

Ein entscheidender Punkt, der oft Missverständnisse verursacht: Das Modul überträgt ausschließlich Daten, nicht Code. Eine DataClass kann Methoden haben - berechnete Properties, Validierungs-Logik, Utility-Funktionen - aber diese Methoden existieren nur dort, wo die DataClass-Definition im Code vorhanden ist. Sie werden nicht über Shared Memory übertragen und sind auf der Reader-Seite nicht verfügbar, es sei denn der Reader importiert die gleiche DataClass-Definition.

Betrachten wir eine DataClass mit Methoden:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class RobotState:
    position: "float64[3]"
    velocity: "float64[3]"
    timestamp: float
    
    def get_speed(self) -> float:
        """Berechne Geschwindigkeit aus Velocity-Vektor."""
        return np.linalg.norm(self.velocity.value)
    
    def is_moving(self) -> bool:
        """Prüfe ob Roboter sich bewegt."""
        return self.get_speed() > 0.01
    
    def age_seconds(self, current_time: float) -> float:
        """Wie alt sind die Daten?"""
        return current_time - self.timestamp.value
```

Wenn der Writer diese DataClass verwendet, stehen ihm alle Methoden zur Verfügung. Er kann `state.get_speed()` aufrufen, `state.is_moving()` prüfen, und so weiter. Aber was passiert auf der Reader-Seite?

Der Reader hat zwei Möglichkeiten, auf die Daten zuzugreifen:

**Fall 1: Reader ohne DataClass-Import (Auto-Reconstruction)**

Wenn der Reader nur mit dem Namen attachiert - `shm = SharedMemory(name)` ohne `expected_type` - rekonstruiert das Modul die DataClass dynamisch. Diese rekonstruierte Klasse wird mit `make_dataclass()` erstellt und enthält ausschließlich die Feld-Definitionen:

```python
# Was das Modul intern macht (vereinfacht):
from dataclasses import make_dataclass

ReconstructedClass = make_dataclass(
    'DynamicDataClass_shm_abc123',
    [
        ('position', "float64[3]"),
        ('velocity', "float64[3]"),
        ('timestamp', float)
    ]
)
```

Diese dynamisch erstellte Klasse hat **keine** der Methoden der Original-Klasse. Sie hat nur:
- Die drei Felder (`position`, `velocity`, `timestamp`)
- Die von `@dataclass` automatisch generierten Standard-Methoden (`__init__`, `__repr__`, `__eq__`)
- Keine `get_speed()`, keine `is_moving()`, keine `age_seconds()`

Wenn der Reader versucht, `data.get_speed()` aufzurufen, bekommt er `AttributeError: 'DynamicDataClass_shm_abc123' object has no attribute 'get_speed'`. Die Methode existiert schlicht nicht, weil sie nicht aus dem Header rekonstruiert werden konnte. Im Header stehen nur Feld-Namen, Typen, und Offsets - keine Methoden-Definitionen, kein Bytecode, keine Logik.

**Fall 2: Reader mit DataClass-Import (Validation)**

Wenn der Reader die Original-DataClass importiert und beim Attach angibt - `shm = SharedMemory(name, expected_type=RobotState)` - passiert folgendes: Das Modul liest den Header, rekonstruiert die Struktur zur Validierung, vergleicht sie mit `RobotState`, und wenn sie übereinstimmt, verwendet es `RobotState` als DataClass-Typ für die zurückgegebenen Objekte.

In diesem Fall **hat** die zurückgegebene Instanz alle Methoden:

```python
from robot_module import RobotState  # Import der Original-Definition

shm = SharedMemory(name, expected_type=RobotState)
data = shm.read()

# Jetzt funktioniert:
speed = data.get_speed()
if data.is_moving():
    print(f"Robot moving at {speed} m/s")
```

Aber - und das ist wichtig - die Methoden sind **lokaler Code**, nicht übertragene Funktionalität. Der Reader muss die gleiche Python-Datei haben, die die Methoden definiert. Wenn die Writer-Version der Klasse `get_speed()` anders implementiert als die Reader-Version, führt das zu inkonsistentem Verhalten. Die Daten kommen aus Shared Memory, aber die Methoden sind lokaler Code, der auf diese Daten operiert.

## Warum keine Code-Übertragung?

Man könnte fragen: Warum überträgt das Modul nicht auch die Methoden? Python's `pickle` kann schließlich Funktionen serialisieren. Die Antwort ist mehrschichtig:

**Sicherheit:** Code-Übertragung ist ein massives Sicherheitsrisiko. Wenn der Writer beliebigen Code in Shared Memory schreiben könnte, den der Reader dann ausführt, öffnet das die Tür für Remote Code Execution. Ein kompromittierter oder böswilliger Writer könnte schädlichen Code einschleusen. Bei reinen Daten ist das unmöglich - Daten können falsch oder korrumpiert sein, aber sie können nicht "ausgeführt" werden.

**Versions-Konflikte:** Code ändert sich. Die Writer-Version einer Methode könnte Bug-Fixes haben, die die Reader-Version nicht hat. Oder umgekehrt. Überträgt man Code, muss man Versionierung implementieren - welche Version der Methode gilt? Wer entscheidet? Bei reinen Daten ist die Struktur der "Contract", und solange die Struktur passt, funktioniert alles.

**Komplexität:** Methoden können beliebig komplex sein. Sie können auf externe Module zugreifen, Dateien öffnen, Netzwerk-Requests machen. Will man das übertragen, muss man nicht nur den Bytecode serialisieren, sondern auch alle Abhängigkeiten. Das wird schnell unpraktikabel. Ein DataClass-Feld ist einfach - ein Name, ein Typ, ein Offset. Eine Methode ist ein ganzer Dependency-Graph.

**Performance:** Methoden-Übertragung würde den Header massiv aufblähen. Bytecode für mehrere Methoden könnte Kilobytes sein. Der Header, der jetzt vielleicht 200 Bytes ist, würde auf 10 KB wachsen. Das muss bei jedem Attach gelesen und geparst werden - ein Performance-Hit ohne klaren Nutzen.

Die Design-Philosophie ist: **Shared Memory ist für Daten, nicht für Logik.** Logik bleibt in den Prozessen, Daten fließen zwischen ihnen. Dies ist eine saubere Trennung, die Komplexität begrenzt und Sicherheit erhöht.

## Praktische Implikationen und Workarounds

Für den Entwickler bedeutet dies: Methoden in DataClasses funktionieren nur, wenn beide Seiten die DataClass-Definition importieren. Dies hat mehrere Konsequenzen:

**Best Practice 1: Shared Module**
Die DataClass sollte in einem gemeinsamen Modul definiert sein, das sowohl Writer als auch Reader importieren:

```python
# shared_types.py
from dataclasses import dataclass

@dataclass
class SensorData:
    temperature: float
    pressure: float
    
    def is_valid_range(self) -> bool:
        return -50 < self.temperature.value < 150

# writer.py
from shared_types import SensorData
shm = SharedMemory(SensorData)

# reader.py
from shared_types import SensorData
shm = SharedMemory(name, expected_type=SensorData)
data = shm.read()
if data.temperature.valid and data.is_valid_range():
    process(data)
```

**Best Practice 2: Utility-Funktionen statt Methoden**
Für Fälle, wo man nicht sicher ist, ob der Reader importieren kann, verwendet man freie Funktionen statt Methoden:

```python
# shared_types.py
@dataclass
class SensorData:
    temperature: float
    pressure: float

def is_valid_range(data: SensorData) -> bool:
    """Utility-Funktion, die DataClass als Parameter nimmt."""
    return -50 < data.temperature.value < 150

# reader.py (kann auch ohne Import von shared_types funktionieren)
shm = SharedMemory(name)  # Auto-reconstruction
data = shm.read()

# Wenn man die Utility-Funktion braucht, importiert man sie:
from shared_types import is_valid_range
if is_valid_range(data):
    process(data)
```

**Best Practice 3: Minimale Methoden**
DataClasses für Shared Memory sollten idealer Weise gar keine Methoden haben, oder nur sehr einfache, die leicht nachzubauen sind. Die DataClass ist primär ein Daten-Container, keine Business-Logic-Klasse. Komplexe Berechnungen gehören in separate Service-Klassen oder Module.

```python
# Gut: Reine Daten
@dataclass
class Point3D:
    x: float
    y: float
    z: float

# Weniger gut: Viel Logik
@dataclass
class Point3D:
    x: float
    y: float
    z: float
    
    def distance_to(self, other): ...
    def normalize(self): ...
    def rotate(self, angle): ...
    # ... 10 weitere Methoden
```

Für komplexe Punkt-Operationen wäre eine separate `Point3DOperations`-Klasse oder ein `geometry`-Modul besser. Die DataClass bleibt schlank und fokussiert auf Daten.

## Was der Reader tatsächlich bekommt

Wenn der Reader `data = shm.read()` aufruft, bekommt er eine DataClass-Instanz, deren Felder **nicht** die rohen Werte sind, sondern `ValueWithStatus`-Wrapper. Dies ist unabhängig davon, ob die DataClass-Definition importiert wurde oder auto-rekonstruiert wurde.

```python
data = shm.read()
type(data)                    # → RobotState (oder DynamicDataClass_...)
type(data.position)           # → ValueWithStatus
type(data.position.value)     # → numpy.ndarray (float64[3])
```

Diese Wrapper-Objekte haben ihre eigenen Properties und Methoden:
- `data.position.value` - Der eigentliche Wert
- `data.position.valid` - Boolean Property
- `data.position.modified` - Boolean Property
- `data.position.truncated` - Boolean Property
- `data.position.unwritten` - Boolean Property
- `data.position.overflow` - Boolean Property (FIFO)

Zusätzlich implementieren die Wrapper Magic Methods für Convenience:
- `float(data.temperature)` - Konvertierung zu Float
- `data.temperature + 5` - Arithmetik
- `np.array(data.position)` - NumPy-Integration

Aber die Wrapper sind **nicht** die Original-Werte. Sie sind Proxy-Objekte, die den Wert und seinen Status kapseln. Dies ist wichtig zu verstehen: Selbst wenn der Reader die Original-DataClass importiert hat, sind die Feld-Werte keine rohen Floats oder Arrays, sondern Wrapper. Man greift über `.value` auf den eigentlichen Wert zu, oder nutzt die Magic Methods für transparenten Zugriff.

## Zusammenfassung: Code vs Daten

Die Kern-Regel: **Shared Memory überträgt Struktur und Daten, nicht Verhalten.** Der Header enthält genug Information, um das Memory-Layout zu verstehen und die Felder zu rekonstruieren. Er enthält keine Information über Methoden, Logik, oder Verhalten.

Wenn beide Prozesse die gleiche DataClass-Definition haben (durch gemeinsamen Import), funktionieren Methoden wie erwartet - aber sie sind nicht "übertragen" worden, sondern existieren unabhängig in beiden Prozess-Images. Wenn die Prozesse unterschiedliche Versionen oder Definitionen haben, divergiert das Verhalten - aber die Daten bleiben konsistent, weil die Struktur im Header fixiert ist.

Dies ist ein Vorteil, kein Nachteil: Es zwingt zu sauberer Trennung zwischen Daten (was geteilt wird) und Logik (was lokal bleibt). Es verhindert Code-Injections und Versions-Hölle. Und es hält das Modul einfach, fokussiert, und performant. Für Inter-Process-Communication ist Daten-Übertragung das Wesentliche - Logik kann jeder Prozess selbst haben.

--------------------------------------------------------------