# 50-Step Plan: Hardcore Physics Engine Expansion

Ziel: Die bestehende Cellular-Automata-Physik schrittweise durch echte,
mathematisch fundierte Kontinuumsmechanik ersetzen bzw. ergänzen.
Endzustand: ein hybrides CA/PDE-System das Navier-Stokes-Fluiddynamik,
Wärmeleitung (Fourier), Diffusion (Fick), Druckwellen und Strukturmechanik
auf dem gleichen Grid vereint — bei vertretbarer Framerate auf einem Pi.

---

## Phase 1 — Fundament & Datenstrukturen (Schritte 1–8)

### Schritt 1 — Float-Grid-Refactor
Alle physikalischen Felder (Temperatur, Druck, Dichte, Geschwindigkeit) von
Listen-of-Lists auf `numpy`-Arrays (`float32`) migrieren.  
Vorteil: vektorisierte Operationen, ~50× schneller als Python-Loops.  
Felder: `temperature[R,C]`, `pressure[R,C]`, `density[R,C]`,
`vel_x[R,C]`, `vel_y[R,C]`, `oxygen[R,C]`, `smoke[R,C]`.

### Schritt 2 — Materialdichte-Feld
Dynamisches `density_field[R,C]` das aus `grid[R,C]` + Materialparametern
+ Temperaturausdehnung berechnet wird. Basis für Auftrieb und Druckgradient.

### Schritt 3 — Druckfeld einführen
Skalares `pressure[R,C]` (Pascal). Initialisierung: hydrostatisch
`p = ρ·g·h`. Wird jedes Substep aktualisiert. Basis für Step 9 (Poisson).

### Schritt 4 — Geschwindigkeitsfeld einführen
`vel_x[R,C]`, `vel_y[R,C]` (m/s, skaliert auf Grid-Einheiten).
Advektionsschritt bewegt Felder entlang dieses Vektorfelds.

### Schritt 5 — Substep-Architektur
Physik-Loop in explizite Substeps aufteilen, die in fester Reihenfolge laufen:
1. External forces  2. Pressure solve  3. Velocity advect
4. Scalar advect    5. Diffusion       6. Phase change
7. Chemistry        8. CA-fallback     9. Boundary conditions

### Schritt 6 — Boundary-Condition-System
Klasse `BoundaryConditions` mit konfigurierbaren Randbedingungen:
- `no-slip` (Wand), `free-slip`, `open` (Luft strömt aus),
- `inlet` (definierter Einströmquerschnitt), `outlet`.

### Schritt 7 — Grid-Koordinatensystem
Physikalische Einheiten: 1 Zelle = 0.05 m, 1 Tick = 1/60 s.
`dt`, `dx`, `dy` als Konstanten. Courant-Zahl CFL = `max(|u|)·dt/dx` < 0.5
wird jedes Frame geprüft; bei Überschreitung wird `dt` halbiert (adaptive).

### Schritt 8 — Profiler-Integration
Jeder Substep misst seine Laufzeit (`time.perf_counter`). HUD zeigt
`ms/step` per Substep. Basis für spätere numpy→numba-Migration.

---

## Phase 2 — Navier-Stokes Fluidsolver (Schritte 9–18)

### Schritt 9 — Poisson-Drucklöser (Jacobi)
Inkompressibilitätsbedingung `∇·u = 0` erzwungen via Poisson-Gleichung:
`∇²p = (ρ/dt)·∇·u*`
Gelöst mit 20–40 Jacobi-Iterationen pro Substep (ausreichend für visuelle
Qualität). Implementierung vollständig vektorisiert mit numpy roll-shifts.

### Schritt 9b — MAC-Grid Topologie (Staggered Grid)
Anstatt $p$, $u$ und $v$ im selben Zellzentrum (Collocated) zu speichern, muss ein Marker-and-Cell (MAC) Grid verwendet werden. 
* **Architektur:** Druck $p$ sowie skalare Felder (Temperatur, Dichte) liegen im Zellzentrum. Geschwindigkeiten liegen auf den Zellkanten: $u_{i+1/2, j}$ und $v_{i, j+1/2}$.
* **Warum:** Verhindert das "Checkerboard-Problem" (Odd-Even-Decoupling) in der Poisson-Gleichung, bei dem hochfrequente Druckoszillationen vom Gradienten-Operator nicht "gesehen" werden. Absoluter Standard für stabile, inkompressible N-S-Solver.

### Schritt 10 — Druckgradient-Korrektur
Velocity-Update nach Druck-Solve:
`u = u* - (dt/ρ)·∇p`
Stellt Divergenzfreiheit sicher. Separiere in x- und y-Komponente.

### Schritt 11 — Viskosität (implizite Diffusion)
Viskosen Term `μ·∇²u` implizit lösen (Crank-Nicolson) um Stabilitätsprobleme
bei hoher Viskosität zu vermeiden. Materialabhängige `μ` aus MATERIALS-Dict.

### Schritt 12 — Semi-Lagrange-Advektion
Rückwärts-Advektion ("trace particle back") für Geschwindigkeit und Skalare:
`q(x) = q(x - u·dt)` mit bilinearer Interpolation.
Stabil für CFL > 1, kein explizites Stabilitätslimit.

### Schritt 12b — BFECC-Advektion (MacCormack)
Reines Semi-Lagrange ist zwar bedingungslos stabil (CFL > 1), besitzt aber eine massive numerische Diffusion – Wirbel sterben extrem schnell ab, VOF-Grenzflächen verschmieren.
* **Erweiterung:** Back and Forth Error Compensation and Correction (BFECC) oder MacCormack-Advektion.
* **Ablauf:** 1. Advektiere Feld $q$ vorwärts: $\hat{q} = \text{advect}(q, \mathbf{u}, dt)$
  2. Advektiere $\hat{q}$ rückwärts: $\check{q} = \text{advect}(\hat{q}, -\mathbf{u}, dt)$
  3. Berechne Fehler: $e = (q - \check{q}) / 2$
  4. Finale Advektion mit kompensiertem Startfeld: $q^{n+1} = \text{advect}(q + e, \mathbf{u}, dt)$
* **Ergebnis:** Erhält hochfrequente Details und Vorticity drastisch besser. Kostet $3\times$ mehr Advektions-Lookups, was dank `numba`/`numpy` aber völlig im Budget liegt.

### Schritt 13 — Auftriebskraft (Boussinesq)
Thermischer Auftrieb: `F_y = -g·β·(T - T_ref)·ρ`
Dichte-Auftrieb: `F_y = -g·(ρ - ρ_ref) / ρ_ref`
Beide Terme addiert zur externen Kraft vor dem Pressure-Solve.

### Schritt 14 — Vorticity Confinement
Kleine Wirbelstrukturen die durch numerische Diffusion verloren gehen
werden künstlich verstärkt: `F = ε·(η × ∇|ω|)`
Parameter `ε` im Config-Profil einstellbar. Macht Rauch und Flammen
visuell realistischer ohne Gitterauflösung erhöhen zu müssen.

### Schritt 15 — Freie Oberflächen (VOF-light)
Volume-of-Fluid-ähnliche Flüssigkeitsoberfläche: `fill_level[R,C] ∈ [0,1]`
Zellen können teilweise befüllt sein. Oberfläche wird korrekt gerendert.
Notwendig für realistische Wasserstrahl- und Überschwemmungssimulation.

### Schritt 15b — Level-Set Methode & CSF (Oberflächenspannung)
Dein "VOF-light" (Schritt 15) wird durch die Advektion sofort in einen unscharfen Gradienten zerlaufen. 
* **Level-Set ($\phi$):** Statt eines reinen Fill-Levels wird ein Signed Distance Field (SDF) mitgeführt. $\phi < 0$ ist Flüssigkeit, $\phi > 0$ ist Gas, $\phi = 0$ ist die exakte, gestochen scharfe Oberfläche. Re-Initialisierung via Fast Marching oder Eikonal-Gleichung hält den Gradienten bei $|\nabla\phi| = 1$.
* **CSF (Continuum Surface Force):** Ermöglicht Wassertropfen und Kapillareffekte. Die Oberflächenkrümmung $\kappa$ wird direkt aus dem Level-Set berechnet: $\kappa = \nabla \cdot (\frac{\nabla\phi}{|\nabla\phi|})$. Die Kraft wird als Quellterm in den Impulsgleichungen angewendet: $\mathbf{F}_{surf} = \sigma \kappa \delta(\phi) \nabla\phi$.

### Schritt 16 — Turbulenzmodell (Smagorinsky-LES)
Sub-Grid-Scale Turbulenz: effektive Viskosität erhöht sich mit lokalem
Schergradienten: `μ_eff = ρ·(C_s·Δx)²·|S|`
Aktivierbar in "realistic"-Profil. Macht große Reynoldszahlen stabil.

### Schritt 17 — Mehrphasen-Dichte-Kopplung
Flüssigkeiten unterschiedlicher Dichte (Wasser ρ=1000, Öl ρ=850, Lava ρ=2800)
treiben echten Druckgradienten statt CA-Density-Swap.
Rayleigh-Taylor-Instabilität entsteht automatisch.

### Schritt 17b — Immersed Boundary Method (Fractional Solid Coupling)
Die Brücke zwischen deiner CA-Pulver-Physik und dem PDE-Strömungslöser. Wenn Sand in Wasser fällt, muss das Wasser verdrängt werden.
* **Mechanik:** Jede Zelle führt ein Feststoff-Fraktionsfeld $V_{frac} \in [0,1]$.
* **Kopplung:** Die Inkompressibilitätsbedingung im Poisson-Solver (Schritt 9) wird modifiziert. Anstatt $\nabla \cdot \mathbf{u}^* = 0$ zu erzwingen, muss die Divergenz die Bewegung der Feststoffe kompensieren: $\nabla \cdot \mathbf{u}^* = \nabla \cdot (\mathbf{u}_{solid} V_{frac})$. 
* **Effekt:** Verhindert, dass Flüssigkeit durch CA-Feststoffe (wie fallenden Sand oder kollabierende Wände) vernichtet oder aus dem Nichts generiert wird. Erzeugt korrekten hydrodynamischen Staudruck vor bewegten Objekten.

### Schritt 18 — Gasphysik (kompressibel, vereinfacht)
Gase (Rauch, Feuer, Plasma) folgen idealem Gasgesetz: `p·V = n·R·T`
Lokale Druckerhöhung bei Verbrennung → Expansion → reale Druckwelle.

---
## Phase 3 — Wärme & Diffusion (Schritte 19–25)

### Schritt 19 — Fourier-Wärmeleitungsgleichung (ADI)
$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$ mit materialabhängiger Thermaldiffusivität $\alpha = \frac{k}{\rho c_p}$. 
Implizite ADI-Methode (Alternating Direction Implicit) für unbedingten Stabilitätsbereich. Ersetzt iterativen CA-Thermal-Step.
**Erweiterung (Harmonisches Mittel):** Um physikalisch korrekte Wärmeflüsse an extremen Materialgrenzen (z.B. Stahl zu Luft, $\Delta k \approx 10^4$) zu garantieren, wird die thermische Leitfähigkeit an den Zellkanten zwingend als harmonisches Mittel berechnet: $k_{i+1/2} = \frac{2 k_i k_{i+1}}{k_i + k_{i+1}}$. Dies verhindert numerische Instabilitäten und überhöhte Interface-Temperaturen.

### Schritt 20 — Konvektiver Wärmetransport
$\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T = \alpha \nabla^2 T$
Wärme bewegt sich mit dem Strömungsfeld (Semi-Lagrange, wie in Phase 2 definiert). Thermische Auftriebsschleifen entstehen automatisch aus der dynamischen Kopplung mit Schritt 13.

### Schritt 21 — Strahlungstransport (RTE) & Partizipierende Medien
Einfaches Ray-Marching reicht für CFD nicht aus, da Gase Strahlung absorbieren. Implementierung einer vereinfachten Strahlungstransportgleichung (RTE) via Discrete Ordinates Method (DOM) in 8 Richtungen.
* **Emission:** Heiße Feststoffe und Ruß emittieren gemäß Stefan-Boltzmann $q_{rad} = \varepsilon \sigma T^4$.
* **Absorption:** Das `soot`-Feld (Ruß) und `h2o`-Feld (Wasserdampf) agieren als *partizipierende Medien*. Sie absorbieren Infrarotstrahlung auf dem Strahlweg ($dI/ds = -\kappa I + \kappa I_b$), heizen sich dabei selbst auf und blockieren die Strahlung für dahinterliegende Zellen (Sichtverschattung bei Gebäudebränden).

### Schritt 22 — Phasenübergänge (Enthalpie-Porositäts-Methode)
Bei Schmelzen/Erstarren/Verdampfen: Energie wird als latente Wärme $Q = m L$ absorbiert bzw. freigesetzt.
Zur numerischen Stabilisierung des Stefan-Problems wird die Enthalpie-Porositäts-Methode genutzt: Die Temperatur ist eine Funktion der Gesamtenthalpie $H$. Im Phasenübergangsbereich existiert eine "Mushy Zone" (teigiger Zustand). Die Feststoff-Fraktion $f_s \in [0,1]$ skaliert die Viskosität im Navier-Stokes-Solver lokal gegen unendlich, was die Strömung beim Erstarren physikalisch korrekt zum Erliegen bringt.

### Schritt 23 — Gasgemische, Fick'sche Diffusion & Molare Dichte

**Initiale Luftzusammensetzung** (Volumenprozent, Normal-Atmosphäre):

| Gas | Anteil | Modelliert als | Molare Masse $M_i$ |
|---|---|---|---|
| N₂ | 78.09 % | inert, dominanter Wärmeträger | 28.01 g/mol |
| O₂ | 20.95 % | reaktiv, `o2[R,C]` | 32.00 g/mol |
| Ar | 0.93 % | → N₂ zugeschlagen (inert) | - |
| CO₂ | 0.04 % | `co2[R,C]`, Startwert | 44.01 g/mol |
| H₂O | ~1 % | `h2o[R,C]`, abhängig von r.F. | 18.02 g/mol |

N₂ wird **nicht** als eigenes Feld gespeichert — seine Konzentration ergibt sich implizit: $X_{N_2} = 1 - X_{O_2} - X_{CO} - X_{CO_2} - X_{H_2O} - X_{soot}$.
N₂ dient als **Wärmesenke** in Verbrennungsreaktionen (senkt adiabate Flammentemperatur) und als **Verdünnungsmittel** (Erstickung ab O₂ < 15 %).

**Fick'sche Diffusion** für jeden reaktiven Gasanteil:
$\frac{\partial c}{\partial t} = D \nabla^2 c$
Unterschiedliche Diffusionskoeffizienten: $D_{O_2} = 2.0\times10^{-5} \text{ m}^2/\text{s}$, temperaturabhängig: $D \propto T^{1.75}$.

**Erweiterung (Molar Mass Buoyancy):** Das Gemisch berechnet in jedem Frame seine lokale molare Masse: $M_{mix} = \sum X_i M_i$. 
Die Gasdichte für den hydrostatischen Druck (Schritt 3) und den Auftriebsterm (Schritt 13) wird direkt über das ideale Gasgesetz ermittelt: $\rho = \frac{p M_{mix}}{R T}$. Dadurch sinkt schweres kaltes CO₂ physikalisch korrekt zu Boden, während leichter, heißer Wasserdampf extrem schnell aufsteigt.

### Schritt 24 — Poröser Feuchtigkeitstransport (Darcy/Richards)
`moisture[R,C] \in [0,1]`. Feuchte Materialien benötigen mehr Energie zum Zünden (Wärmekapazität des Wassers + Verdampfungsenthalpie). 
**Interne Dynamik:** Feuchtigkeit in Feststoffen (`wood`, `stone`) diffundiert nicht einfach linear, sondern folgt dem kapillaren Transport (vereinfachte Richards-Gleichung). Wasser zieht durch Kapillarkräfte in trockene angrenzende Feststoff-Zellen ein. An der Grenze zu Gaszellen verdunstet es in das `h2o`-Feld (Rate abhängig von lokaler Luftfeuchtigkeit und Temperatur).

### Schritt 25 — Verdunstungskühlung & Leidenfrost-Effekt
Wenn flüssiges Wasser auf heiße Feststoffoberflächen trifft: Energie $Q = m L_v$ wird entzogen, massiver Dampfstoß nach oben. 
**Erweiterung (Film Boiling):** Implementierung des Leidenfrost-Effekts. Liegt die Oberflächentemperatur signifikant über der Siedetemperatur ($T_{surf} > 200\,^\circ\text{C}$), bildet sich sofort eine isolierende Dampfschicht im MAC-Grid. Der Wärmeübergangskoeffizient ($h$) kollabiert drastisch. Das Wasser verdampft *langsamer* und fließt als Tropfen über das heiße Material ab, anstatt es direkt zu löschen. Essenziell für hochpräzise Feuerwehr-Szenarien.

---

## Phase 4 — Druck, Schall & Explosionen (Schritte 26–31)

### Schritt 26 — Akustische Wellenausbreitung & PML (Perfectly Matched Layers)
Separates akustisches Sub-Stepping für Druckwellen $dp[R,C]$ und akustische Partikelgeschwindigkeit $\mathbf{u}_{ac}$. Da Schallwellen das reguläre CFL-Limit des Fluid-Solvers sprengen, wird die Wellengleichung explizit mit einem akustischen Zeitschritt $dt_{ac} \ll dt$ gelöst.
**Erweiterung (PML):** Um zu verhindern, dass Explosionen und Schallwellen als physikalisch inkorrekte Echos von den Rändern des Simulationsgebiets endlos zurückprallen, werden die Gitterränder mit *Perfectly Matched Layers* (PML) ausgestattet. Diese dämpfen die Wellengleichung an den Rändern exponentiell ab, was einen unendlichen offenen Raum simuliert.

### Schritt 27 — Detonationskinetik & Divergenz-Injektion (Sedov-Taylor)
Statt eine Explosion als simplen radialen Impuls-Hack (Push) abzuhandeln, wird sie tief in den Navier-Stokes-Solver integriert.
Wenn eine Zelle mit extrem hoher Rate verbrennt (z.B. Gunpowder, C4), wird die resultierende Gasexpansion als massiver Quellterm in die rechte Seite der Poisson-Druckgleichung injiziert: $\nabla \cdot \mathbf{u}^* = \frac{\dot{Q}}{\rho h_{cell}}$.
**Effekt:** Der Jacobi/Multigrid-Solver generiert im nächsten Substep automatisch ein radialsymmetrisches, massives Druckfeld. Eine Sedov-Taylor-Blast-Wave entsteht emergent durch die Navier-Stokes-Gleichungen, mitsamt korrektem Unterdruck-Sog (Negative Phase) im Zentrum *nach* der Explosion.

### Schritt 28 — Schockwellen-Versagen & Spallation
Druckinduziertes Zerbrechen nutzt nicht mehr einen simplen statischen $\Delta p$-Vergleich.
Die Zerstörungskraft auf Feststoffe wird exakt aus dem Druckgradienten des MAC-Grids abgeleitet: $\mathbf{F}_{shock} = -\nabla p$. Übersteigt diese Kraft (kombiniert mit der Scherbelastung aus dem Strömungsfeld) den dynamischen Fließdruck (Yield Stress, $\sigma_y$) des Materials nach dem von-Mises-Kriterium, bricht die Zelle.
**Erweiterung (Spallation):** Trifft eine starke Druckwelle auf eine harte Wand und wird reflektiert, interferieren einfallende und reflektierte Welle. Durch die Zugspannung auf der Rückseite platzen physikalisch korrekt Debris-Partikel ab, selbst wenn die Wand nicht komplett durchbrochen wird.

### Schritt 29 — Emergenter Hydrostatischer Druck & Free-Surface BC
Der Torricelli-Ansatz ($v = \sqrt{2gh}$) wird komplett entfernt, da er für einen inkompressiblen Solver ein numerischer Hack ist.
**Upgrade:** Hydrostatischer Druck entsteht stattdessen *völlig nativ*. Die Schwerkraft $\mathbf{g}$ wird im Advektionsschritt als Kraftfeld auf die Flüssigkeit addiert ($\mathbf{u}^* = \mathbf{u}^n + \mathbf{g} dt$). Der entscheidende Faktor: An der Level-Set-Oberfläche ($\phi = 0$) wird strikt die Dirichlet-Randbedingung $p = 0$ (Atmosphärendruck) erzwungen. Die Poisson-Lösung $\nabla^2 p = \nabla \cdot \mathbf{u}^*$ liefert daraufhin automatisch das exakte hydrostatische Feld $p = \rho g h$ in der Tiefe. Strömungen aus Öffnungen ergeben sich daraus zwingend nach Bernoulli.

### Schritt 30 — Atmosphärische Grenzschicht (Log-Wind-Profil)
Umgebungsströmungen sind keine simplen Vektoradditionen mehr. Externe Winde werden über feste Dirichlet-Randbedingungen an den MAC-Grid-Kanten (Einström-/Ausströmränder) eingespeist.
**Erweiterung:** Bei offenen Bodenszenarien wird ein logarithmisches Windprofil für die Randbedingung angelegt: $u(z) = \frac{u_*}{\kappa} \ln(\frac{z}{z_0})$. Dies zwingt die Strömung über dem Boden physikalisch in Scherung, was turbulente Grenzschichten (Boundary Layers) erzeugt und Rauch/Gase hochrealistisch über Terrain-Unebenheiten verwirbelt.

### Schritt 31 — Poröse Medien (Darcy-Forchheimer-Gleichung)
Statt Fenster und Türen über abstrakte `permeability`-Timer abzuwickeln, werden teildurchlässige Strukturen als poröse Medien direkt in die N-S-Impulsgleichung eingebettet.
Das Material erhält Permeabilität $K$ und Formfaktor $C_F$. Ein Senkenterm wird der Geschwindigkeit aufgeprägt: $\mathbf{F}_{drag} = -\left( \frac{\mu}{K}\mathbf{u} + \frac{C_F}{\sqrt{K}}\rho|\mathbf{u}|\mathbf{u} \right)$.
**Effekt:** Ein geschlossenes, aber minimal undichtes Fenster erlaubt einen langsamen Darcy-Fluss, während eine zerbrochene Tür turbulente Strömungen dämpft. Der O2-Einstrom für einen Backdraft baut sich extrem präzise abhängig von den Druckunterschieden $\Delta p$ zwischen Innen- und Außenraum auf.

---

## Phase 5 — Strukturmechanik & FEM-Light (Schritte 32–37)

### Schritt 32 — Cauchy-Spannungstensor & Verzerrung (Strain)
Der simple skalare `strength`-Wert wird durch eine vollständige tensoriell-mechanische Formulierung ersetzt. 
Jede Feststoff-Zelle führt ein Verschiebungsfeld $\mathbf{d}[R,C]$ und einen symmetrischen $2\times2$ Cauchy-Spannungstensor $\boldsymbol{\sigma} = \begin{pmatrix} \sigma_{xx} & \tau_{xy} \\ \tau_{xy} & \sigma_{yy} \end{pmatrix}$.
Die Verzerrung (Strain) $\boldsymbol{\varepsilon}$ ergibt sich aus dem symmetrischen Gradienten der Verschiebung: $\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{d} + (\nabla \mathbf{d})^T)$. Über das verallgemeinerte Hookesche Gesetz für isotrope Materialien ($\boldsymbol{\sigma} = 2\mu\boldsymbol{\varepsilon} + \lambda \text{tr}(\boldsymbol{\varepsilon})\mathbf{I}$ mit den Lamé-Parametern $\mu, \lambda$) wird die Spannung exakt berechnet. Dies ermöglicht die korrekte Abbildung von Scherkräften und Torsion in Bögen oder Trägern.

### Schritt 33 — Thermo-Elastoplastizität & Degradation
Wärmeausdehnung ist nicht nur eine Kraft, sondern eine Dehnungs-Komponente.
Die Gesamtdehnung addiert sich additiv: $\boldsymbol{\varepsilon}_{total} = \boldsymbol{\varepsilon}_{elastisch} + \boldsymbol{\varepsilon}_{plastisch} + \boldsymbol{\varepsilon}_{thermisch}$.
Die thermische Dehnung ist $\boldsymbol{\varepsilon}_{th} = \alpha \Delta T \mathbf{I}$.
**Erweiterung (Thermische Degradation):** Der Elastizitätsmodul $E(T)$ und die Fließgrenze $\sigma_y(T)$ sind nun Funktionen der Temperatur. Stahl verliert ab 500 °C dramatisch an Tragfähigkeit. Erhitzt sich eine tragende Zelle, sinkt ihre Festigkeit, bis die reguläre statische Last zum Fließen oder Brechen führt (Feuerwehr-Szenario: Einsturzgefahr).

### Schritt 34 — Gitter-Statik & Versagenskriterien (Bruchmechanik)
Statt vereinfachtem Gewichts-Stacking wird die Bewegungsgleichung $\rho \frac{\partial^2 \mathbf{d}}{\partial t^2} = \nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{g}$ über einen expliziten Zeitschritt gelöst.
Das Materialversagen wird durch bruchmechanische Kriterien bestimmt:
* **Duktile Materialien (Metalle):** Nutzen die von-Mises-Fließbedingung ($\sigma_{v} = \sqrt{\sigma_{xx}^2 - \sigma_{xx}\sigma_{yy} + \sigma_{yy}^2 + 3\tau_{xy}^2} \geq \sigma_y$). Bei Überschreitung geht das Material in plastisches Fließen über.
* **Spröde Materialien (Stein/Beton):** Nutzen das Mohr-Coulomb- oder Drucker-Prager-Kriterium, da sie extrem druckstabil, aber zugempfindlich sind. 
Bricht eine Zelle, wird ihr Spannungstensor auf Null gesetzt (Schädigungsmechanik: $D=1$) und sie geht in das `debris`-Feld über.

### Schritt 35 — Hyperelastizität & Geometrische Nichtlinearität
Kleine Verschiebungen ("sub-zellular") reichen bei großen Verformungen (Biegung eines Balkens) nicht aus. 
Um bei Rotationen künstliches Volumenwachstum zu verhindern, wird die Co-rotational Formulation oder ein einfaches Neo-Hookean-Materialmodell für elastische Stoffe wie Gummi oder heiße Kunststoffe implementiert. Das Verschiebungsfeld $\mathbf{d}$ advektiert das Material visuell auf dem Bildschirm, bevor es zum tatsächlichen Bruch kommt. Man *sieht* den Stahlträger durchhängen, bevor er reißt.

### Schritt 36 — Gekoppelte DEM-Kinetik (Discrete Element Method) für Debris
`debris` (Trümmer) sind nicht einfach Sand. Gebrochene Cluster aus Solid-Zellen werden zu abstrakten Polygon-Partikeln konvertiert.
Diese Partikel werden via DEM (Discrete Element Method) rigoros simuliert: Sie rotieren, besitzen Trägheitsmomente, prallen elastisch voneinander ab (Penalty-Spring-Dashpot-Kollisionsmodell) und koppeln sich bidirektional an den Navier-Stokes-Löser. 
Zweiströmungskopplung: Trümmer verdrängen Fluid (veränderter $V_{frac}$ Term in der Poisson-Gleichung) und Fluid (z.B. eine Schockwelle) beschleunigt Trümmer.

### Schritt 37 — Thermomechanisches Spalling (Abplatzungen)
Brandlast ist primär chemisch (Phase 6), hier wird die *strukturelle* Konsequenz simuliert. 
Beton und Stein unterliegen bei massiver Hitzeeinwirkung einem explosiven Abplatzen (Spalling).
**Mechanik:** Wenn der thermische Gradient $\nabla T$ am Rand einer Betonwand massiv ansteigt, induziert die $\boldsymbol{\varepsilon}_{th}$-Dehnung der heißen Randschicht gewaltige Druckspannungen gegen den noch kalten Kern. Übersteigt die Scherspannung $\tau$ das Scherlimit des Betons (verstärkt durch verdampfende, expandierende Feuchtigkeit in der Zelle, Porendruck $p_{pore}$), bricht die äußere Zell-Schicht explosionsartig ab, wird zu `debris` und legt den kühlen Kern für die weitere Brandlast frei.

---

## Phase 6 — Chemie & Verbrennung (Schritte 38–43)

### Schritt 38 — Stiff-Kinetik & Multi-Step Arrhenius
Ein simples Vorwärts-Euler-Update für $k = A \exp(-E_a/(RT))$ kollabiert numerisch, da chemische Zeitskalen ($10^{-6}$ bis $10^{-9}$ s) extrem viel kleiner sind als die fluiddynamischen Zeitskalen $dt$. 
Die Reaktionsgleichungen bilden ein *stiff ODE system* (steifes Anfangswertproblem).
**Erweiterung:** Implementierung eines impliziten Lösers (z.B. Backward-Differentiation-Formula / CVODE-light) pro Zelle. Die Spezies-Massenbrüche $Y_i$ werden nach $\frac{\partial \rho Y_i}{\partial t} + \nabla \cdot (\rho \mathbf{u} Y_i) = \nabla \cdot (\rho D_i \nabla Y_i) + \dot{\omega}_i$ aktualisiert, wobei der chemische Quellterm $\dot{\omega}_i$ implizit über Newton-Raphson-Iterationen innerhalb des Substeps gelöst wird, um Stabilität zu garantieren.

### Schritt 39 — Eddy Dissipation Concept (EDC) & Mixture Fraction
Stöchiometrie reicht in einem CFD-Grid nicht aus, da Brennstoff und Sauerstoff auf subzellularer Ebene mischen müssen, bevor sie reagieren können.
**Erweiterung:** Die reale Verbrennungsrate wird nicht allein durch die Temperatur limitiert, sondern primär durch die turbulente Durchmischung (Magnussen-Hjertager-Modell). 
Die Rate für den Brennstoffverbrauch wird durch die Mikromischungs-Zeitskala der Turbulenz ($\tau = \frac{k}{\varepsilon}$) dominiert: $\dot{\omega}_{f} = C \rho \frac{1}{\tau} \min\left( Y_f, \frac{Y_{O_2}}{s} \right)$.
Das bedeutet: Egal wie heiß das Gas ist, Feuer entsteht nur an den Rändern von Gasschwaden, an denen der Strömungslöser ausreichend Scherkräfte für die molekulare Mischung berechnet.

### Schritt 40 — Gekoppelte Solid-Gas Pyrolyse (Stefan-Problem)
Die Pyrolysefront ist kein bloßer Timer, sondern eine thermodynamische Randbedingung zwischen Feststoff und Gas (ein Moving-Boundary- bzw. Stefan-Problem).
**Mechanik:** Der Massenfluss des ausgasenden Brennstoffs $\dot{m}^{\prime\prime}$ an der Zellgrenze wird durch die exakte Energiebilanz gesteuert: 
$\dot{q}_{rad}^{\prime\prime} + \dot{q}_{conv}^{\prime\prime} - \dot{q}_{cond}^{\prime\prime} = \dot{m}^{\prime\prime} L_v$
(Eingehende Strahlung + Konvektion aus dem Gas abzüglich der Wärmeleitung in den kühlen Kern des Holzes = Pyrolyserate mal Verdampfungsenthalpie). Das ausgasende Material fügt dem MAC-Grid exakt diesen Massenfluss $\dot{m}^{\prime\prime}$ als Injektions-Randbedingung hinzu, was die Flamme vom Holz wegdrückt (Blowing Effect).

### Schritt 41 — Zwei-Gleichungs-Rußmodell (Soot Kinetik)
Ruß entsteht nicht linear aus Kohlenstoff-Überschuss. Es bedarf eines rigorosen semi-empirischen Modells (z.B. Tesner oder Moss-Brookes).
Zwei Transportgleichungen werden hinzugefügt: Eine für die Ruß-Massenfraktion ($Y_{soot}$) und eine für die normalisierte Ruß-Partikeldichte ($N_{soot}$).
Die Evolutionsträume bestehen aus:
1. **Nucleation** (Keimbildung aus Vorläufergasen wie C2H2).
2. **Surface Growth** (Anlagerung von Kohlenstoff an existierende Partikel).
3. **Coagulation** (Kollision und Verschmelzung von Partikeln).
4. **Oxidation** (Verbrennung des Rußes durch O2 und OH-Radikale).
Zusätzlich fungiert $Y_{soot}$ als dominanter Absorptionskoeffizient $\kappa$ in der Strahlungstransportgleichung (Schritt 21).

### Schritt 42 — Elektrolyt-Thermodynamik & Algebraisches Gleichgewicht
Die Modellierung von Säuren und Basen (pH-Feld) als langsame Differenzialgleichung ist physikalisch falsch, da Ionenreaktionen (wie $H^+ + OH^- \rightleftharpoons H_2O$) nahezu instantan ablaufen.
**Mechanik:** Der Transport von reaktiven Spezies erfolgt konvektiv/diffusiv, aber die Neutralisationsreaktion wird als algebraische Nebenbedingung (Mass-Action Law) auf das Feld erzwungen: $K_w = [H^+][OH^-]$.
Nach jedem Advektions-/Diffusionsschritt wird in jeder Zelle ein lokaler Ausgleichsschritt (Fractional Stepping) ausgeführt, der das Ionenprodukt sofort auf das thermodynamische Gleichgewicht zwingt und die korrespondierende Neutralisationsenthalpie $\Delta H_{neut}$ schlagartig in das Temperaturfeld injiziert.

### Schritt 43 — Heterogene Oberflächenkinetik (Langmuir-Hinshelwood)
Katalyse findet nur an der Grenzfläche zwischen Fluid-Zelle und Solid-Zelle statt (Oberflächenreaktion).
**Mechanik:** Einführung eines Adsorptionsmodells. Die katalytische Zelle trackt den Bedeckungsgrad $\theta_i$ für reaktive Gase. 
Die Reaktionsgeschwindigkeit hängt vom Verhältnis der Diffusionsoberfläche zur kinetischen Rate ab (Damköhler-Zahl $Da$). Bei $Da \gg 1$ ist die Reaktion diffusionslimitiert (der Katalysator brennt jedes Molekül weg, das ihn berührt, es entsteht ein massiver Konzentrationsgradient im Fluid). Bei $Da \ll 1$ ist sie kinetisch limitiert. Exotherme Oberflächenreaktionen heizen den Katalysator auf, bis dieser thermisch zu glühen beginnt.

---

## Phase 7 — Performance & Solver-Qualität (Schritte 44–50)

### Schritt 44 — Numba-JIT, SIMD & Memory Alignment
Ein einfaches `@numba.jit` reicht für High-Performance-CFD nicht.
**Upgrade:** Alle kernkritischen Loops werden mit `@njit(parallel=True, fastmath=True)` kompiliert. 
* **Multithreading:** Äußere Grid-Schleifen werden über `numba.prange` auf alle CPU-Kerne verteilt.
* **SIMD-Vektorisierung:** Um AVX2/AVX-512 Instruktionen der CPU auszulasten, müssen die Numpy-Arrays strikt C-contiguous im Speicher liegen. `fastmath` deaktiviert striktes IEEE-754 NaN-Handling und erlaubt Fused Multiply-Add (FMA), was die FLOP/s in der Advektion und im Jacobi-Löser nahezu verdoppelt.

### Schritt 45 — Block-Structured AMR (Adaptive Mesh Refinement)
Ein simples Quad-Tree-Overlay skaliert schlecht auf CPUs wegen unvorhersehbarer Memory-Access-Pattern.
**Mechanik:** Implementierung von Block-Structured AMR nach Berger-Colella. Das Grid wird in feste Blöcke (z.B. $16\times16$ Zellen) unterteilt. Blöcke mit hohem Fehlerindikator (z.B. $\nabla \rho$ oder $\nabla \cdot \mathbf{u} > \text{tol}$) spawnen Child-Blöcke mit halber Zellgröße ($dx/2, dt/2$). 
**Kritisch:** An den Interfaces zwischen feinem und grobem Gitter muss eine exakte Flusskorrektur (Flux-Matching) stattfinden, ansonsten wird durch Interpolationsfehler Masse oder Wärme aus dem Nichts generiert.

### Schritt 46 — Geometric Multigrid (GMG) Poisson-Löser
Der Jacobi-Solver aus Schritt 9 ($O(N^2)$) wird endgültig zum Flaschenhals, da niederfrequente Druckfehler auf großen Grids ewig zum Konvergieren brauchen.
**Architektur:** Ein vollständiger V-Cycle Geometric Multigrid Solver ($O(N)$ Komplexität).
1. **Smoothing:** 2-3 Iterationen Red-Black Gauss-Seidel auf dem feinen Grid, um hochfrequente Fehler zu eliminieren.
2. **Restriction:** Der Residualfehler $\mathbf{r} = \mathbf{f} - \mathbf{A}\mathbf{x}$ wird über einen Full-Weighting-Operator auf ein gröberes Grid ($2dx$) projiziert.
3. **Prolongation:** Die Grobgitter-Korrektur wird bilinear auf das feine Grid hochinterpoliert und addiert.
Selbst für ein $1024\times1024$ Grid konvergiert der Druckausgleich damit in unter 10 V-Zyklen bis zur Maschinengenauigkeit.

### Schritt 47 — Triple-Buffering & Lock-Free Asynchronität
Der GIL (Global Interpreter Lock) blockiert Python-Threads, Numba umgeht ihn (via `nogil=True`), aber der Datenzugriff zwischen Render-Loop und Physik-Loop zerreißt das Bild (Tearing), wenn gelockt wird.
**Lösung:** Ein Lock-freies Triple-Buffer-System für die Vektorfelder.
* `Buffer A`: Wird aktuell vom Pygame-Render-Thread gelesen.
* `Buffer B`: Zuletzt fertiggestellter Frame (wartet auf Swap).
* `Buffer C`: Wird aktuell vom C/Numba-Physik-Thread überschrieben.
**Zeitliche Interpolation:** Pygame rendert Zwischenframes unbegrenzt schnell (z.B. 144 Hz) durch lineare Interpolation zwischen `Buffer A` und dem vorherigen Zustand, während die Physik strikt bei z.B. 60 Hz rechnet. Absolut flüssiges Bild, unabhängig von der Rechenlast.

### Schritt 48 — Sparse-Block-Solving & Narrow Banding
Ruhige Grid-Bereiche zu berechnen verschwendet Cache-Bandbreite.
* **Sparse Blocks:** Das Array wird in $8\times8$ Macro-Blöcke logisch geclustert (z.B. Morton-Code Z-Order Curve für maximale L1-Cache Lokalität). Enthält ein Block nur Luft mit $|\mathbf{u}| < \epsilon$ und $\Delta T \approx 0$, wird er im PDE-Step komplett übersprungen.
* **Narrow Band Level-Set:** Die rechenintensive Re-Initialisierung der VOF-Flüssigkeitsoberfläche (Schritt 15b) wird nicht global berechnet, sondern nur in einem schmalen Band von $\pm 3$ Zellen um die eigentliche Oberfläche ($\phi = 0$).

### Schritt 49 — Dynamische Solver-Toleranzen (Residual Tracking)
Statt harter Iterationszahlen ($N=25$) steuert sich die Engine selbst aus.
Der Drucklöser bricht ab, sobald die $L_2$-Norm des Residuums $||\nabla \cdot \mathbf{u}||_2 < \epsilon_{target}$ fällt. 
**Adaptive Degradation:** Übersteigt die Physik-Zeit das Frame-Budget (z.B. $>16.6\text{ ms}$), greift ein Fallback: $\epsilon_{target}$ wird fließend vergrößert. Die Flüssigkeit wird vorübergehend minimal kompressibler ("bouncy"), aber das System ruckelt nicht. Features wie Strahlung oder Les-Turbulenz werden via Heuristik nahtlos in der Auflösung reduziert.

### Schritt 50 — Rigorose CI/CFD-Validierung & Fehler-Normen
Keine "Sieht gut aus"-Tests mehr. Jede Code-Änderung läuft gegen analytische Literatur-Lösungen zur Verifikation der Navier-Stokes-Disktretisierung.
Automatisierte Metriken in `physics_validation.json`:
* **Taylor-Green Vortex:** Misst die exakte numerische Dissipationsrate der kinetischen Energie im Grid über Zeit. Identifiziert zu starke künstliche Viskosität.
* **Lid-Driven Cavity ($Re=1000$ & $Re=3200$):** Vergleicht die Geschwindigkeitsprofile durch das Zentrum exakt gegen die Tabellen von Ghia et al. (Maximaler $L_\infty$-Fehler $< 2\%$).
* **Poiseuille-Strömung:** Prüft die korrekte Umsetzung der No-Slip-Boundary-Conditions an den Wänden (muss eine perfekte Parabel ergeben).
* **Stefan-Problem:** Überprüft das Aufschmelzen von Eis zu Wasser unter konstanter Wärmezufuhr gegen die analytische Enthalpie-Lösung.

## Phase 8 — Elektromagnetismus & MHD (Schritte 51–53)

### Schritt 51 — Yee-Grid (FDTD-Solver) & Lichtgeschwindigkeit
Die Berechnung elektromagnetischer Wellen erfordert ein spezielles Gitter-Layout, um numerische Instabilitäten zu vermeiden.
**Architektur:** Implementierung eines Yee-Grids (Finite Difference Time Domain). Hierbei sind die Komponenten des elektrischen Feldes $\mathbf{E}$ und des magnetischen Feldes $\mathbf{B}$ sowohl räumlich als auch zeitlich versetzt angeordnet.
Über die Rotationsoperatoren der Maxwell-Gleichungen propagieren Wellen durch das Gitter:
$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$
$\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$
Da die Lichtgeschwindigkeit $c$ physikalische Zeitschritte im Picosekunden-Bereich erfordern würde, wird für die Simulation eine künstlich reduzierte Wellengeschwindigkeit (Low-Speed-Electromagnetics) oder ein voll-impliziter Solver genutzt, um Stabilität bei $dt$ zu wahren.

### Schritt 52 — Lorentz-Kraft & Joule-Erhitzung (MHD-Kopplung)
In diesem Schritt wird der Elektromagnetismus mit der Fluiddynamik (Phase 2) und der Thermodynamik (Phase 3) bidirektional gekoppelt.
* **Impuls-Kopplung (Lorentz-Kraft):** Wenn ionisierte Fluide (Plasma, flüssiges Metall) durch ein Magnetfeld strömen, wird eine Kraft auf das Fluid ausgeübt: $\mathbf{F}_L = \mathbf{J} \times \mathbf{B}$. Dieser Term wird als Volumenkraft in die Navier-Stokes-Impulsgleichung injiziert. Magnetfelder können somit Strömungen aktiv lenken, bremsen oder in Wirbel zwingen.
* **Thermische Kopplung (Joule-Heating):** Elektrischer Widerstand wandelt Stromfluss direkt in Wärme um. Die Verlustleistung $Q_{joule} = \frac{|\mathbf{J}|^2}{\sigma_e}$ wird als lokaler Quellterm in die Fourier-Wärmeleitungsgleichung eingespeist. Hochspannung lässt Drähte glühen und schmelzen (Zustandsänderung Solid → Liquid).

### Schritt 53 — Plasma-Confinement & Railgun-Kinetik
Jedes Material im `MATERIALS`-Dict erhält die Parameter elektrische Leitfähigkeit $\sigma_e$ und magnetische Permeabilität $\mu$.
* **Magnetisches Confinement:** Durch geschickte Anordnung von stromdurchflossenen Spulen-Pixeln kann ein magnetischer Käfig aufgebaut werden. In Kombination mit Phase 6 (Verbrennung) lassen sich so Fusionsreaktor-Szenarien (Tokamak) simulieren, bei denen das Plasma durch die Lorentz-Kraft stabil in der Luft gehalten wird, ohne die Wände zu berühren.
* **Elektromagnetische Beschleunigung:** Implementierung der Railgun-Physik. Ein hoher Strompuls durch zwei parallele Leiter-Schienen erzeugt ein starkes $\mathbf{B}$-Feld zwischen ihnen. Befindet sich ein leitfähiges `debris`-Teilchen (Projektil) dazwischen, wird es durch die resultierende Lorentz-Kraft massiv beschleunigt und verlässt den Lauf mit ballistischer Flugbahn.

---

## Technologie-Stack

| Schicht | Bibliothek | Zweck |
|---|---|---|
| Arrays | `numpy` (float32) | Alle Felder |
| JIT | `numba` (optional) | Inner Loops ×50 |
| Visualisierung | `pygame` | unverändert |
| Tests | `pytest` | Validierungs-Suite |
| Logging | `rich` + `GameLogger` | bereits vorhanden |
| DB | `psycopg` (optional) | Telemetrie-Logging |

## Reihenfolge-Empfehlung

Phasen in Reihenfolge implementieren — jede Phase baut auf der vorherigen auf.
Zwischen Phasen: Spiel muss lauffähig und rückwärtskompatibel bleiben.
Scripting-API (`api.*`) bleibt während aller Phasen stabil.
