# Knee MRI report labeling rubric v1 (2026-10-13)

Label each knee MRI report using only its text. Reports may be in English, Spanish, Turkish, Greek, German, French, Croatian/Serbian, Dutch or Cyrillic-script languages. Each finding is **study-level presence of the abnormality in the imaged knee**. If a report covers both knees, a finding is present if it is present in either.

## Output

Return one JSON object mapping each `hid` to all 12 findings, spelled exactly:

```json
{"<hid>": {"ACL": {"p": 0.95, "state": "pos"}, "MCL": {"p": 0.05, "state": "not_mentioned"}, "...": {}}}
```

Findings: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`. `state` is one of `pos`, `neg`, `not_mentioned`, `uncertain`; `p` is the probability that the finding is present, a number in [0, 1]. Output JSON only.

## Calibration

| state | p |
|---|---|
| `pos`, explicit and unhedged | 0.90 to 0.97 (0.97 for definite or complete; 0.85 to 0.90 for minimal or low-grade) |
| `neg`, explicitly normal, intact or absent | 0.03 to 0.08 |
| `uncertain` (hedged, equivocal, or a borderline rule below) | 0.30 to 0.60, higher for "probable/likely/suggestive", lower for "possible/cannot exclude/subtle" |
| `not_mentioned` | the prior below |

Not-mentioned priors (most structured reports comment on abnormal structures, so silence usually means absent): ACL 0.05, MCL 0.05, Medial Meniscus 0.10, Lateral Meniscus 0.08, Medial OA 0.10, Lateral OA 0.06, PF OA 0.12, Effusion 0.10, Synovitis 0.06, Baker's 0.05, Contusion 0.05, Fracture 0.03. A summary statement such as "otherwise normal", "ligaments intact" or "menisci normal" makes covered findings `neg`, not `not_mentioned`.

## General rules

- **Negation** applies only to its own scope: "no tear of the MCL; ACL tear" is MCL `neg`, ACL `pos`. Cues: en no, without, absent, not seen, intact; es sin, no se observa, no hay, conservado, íntegro; tr yok, izlenmedi, saptanmadı, görülmedi, olağan, intakt; el δεν, χωρίς, ουδέν, ακέραιος, φυσιολογικ-; de kein, ohne, nicht, unauffällig, intakt; fr pas de, sans, absence de, normal, intact; hr bez, nema, nije, uredan/uredna, očuvan; nl geen, zonder, normaal, intact; cyr нет, без, не выявлено, не определяется, сохранен, интактна.
- **Uncertainty** (possible, probable, suspected, cannot exclude, cannot rule out, suggestive, equivocal; es posible, probable, sospecha, no se descarta; tr olası, şüpheli, ekarte edilemez; el πιθανή, ύποπτ-, δεν αποκλείεται; de V.a., fraglich, möglich, nicht auszuschließen; fr possible, probable, suspicion, ne peut être exclu; hr moguće, suspektno, ne može se isključiti; nl mogelijk, verdacht, niet uit te sluiten; cyr возможно, вероятно, подозрение, не исключается) gives `uncertain`.
- **Degenerative versus tear**: degeneration, mucoid change or intrasubstance signal that does not reach the surface is not a tear. A degenerative tear is a tear.
- **Grading**: any grade of a defined abnormality is positive unless a rule below says otherwise.
- **Post-surgical state**: positive only for a current abnormality. A torn or failed graft, re-tear, or residual tear is positive; an intact reconstruction or a stable post-meniscectomy remnant is `neg` with p 0.15. Hardware or tunnels alone are not fractures.
- **Laterality and compartments**: medial = internal/inner (es interno, de Innen-, fr interne, tr iç, el έσω, hr unutarnji/medijalni, nl mediaal, cyr внутренний/медиальный); lateral = external/outer (externo, Außen-, externe, dış, έξω, vanjski/lateralni, lateraal, наружный/латеральный). "Both menisci" or "bicompartmental" sets each named side.
- Do not infer one finding from another (an ACL tear does not imply a contusion) except as stated.

## Commonly confused pairs (apply strictly)

- **Contusion versus Fracture**: bone bruise or marrow edema without a fracture line is Contusion only (Fracture `neg` 0.05). Fracture needs a cortical or trabecular fracture line or an explicit "fracture" mention (Segond, tibial plateau and osteochondral included). Both described: both positive.
- **OA by compartment**: each OA finding needs evidence in its own compartment. "Tricompartmental" makes all three positive; "degenerative changes" or OA without a compartment is `uncertain` 0.45 for each, never positive for all.
- **Meniscus by side**: score medial and lateral separately; a tear on one side says nothing about the other. Degeneration or signal change not reaching a surface is not a tear.
- **Synovitis versus Effusion**: effusion alone is not synovitis. Synovitis needs synovial thickening, Hoffa fat-pad edema or synovitis wording.

## Findings

1. **ACL**: tear of the anterior cruciate ligament or graft: complete, partial, high-grade, or non-visualised/discontinuous ligament. Mucoid degeneration or ganglion alone is `neg` p 0.15; "sprain/strain" without tear is `uncertain` 0.40. Cues: ACL, anterior cruciate; LCA, ligamento cruzado anterior; ÖÇB, ön çapraz bağ; ΠΧΣ, πρόσθιος χιαστός; VKB, vorderes Kreuzband; LCA, ligament croisé antérieur; LCA/prednji križni ligament; VKB, voorste kruisband; ПКС, передняя крестообразная связка. Tear cues: rotura, desgarro, ruptura; yırtık, kopma; ρήξη; Riss, Ruptur; rupture, déchirure; ruptura, puknuće; scheur, ruptuur; разрыв, повреждение.
2. **MCL**: partial (grade II) or complete (grade III) tear of the medial collateral ligament. Grade I sprain or periligamentous edema only is `uncertain` 0.35. Cues: MCL, LCM, ligamento colateral medial/interno, iç yan bağ (MKL), έσω πλάγιος, Innenband, ligament collatéral médial/interne, medijalni kolateralni ligament, mediale collaterale band, МКС/медиальная коллатеральная связка.
3. **Medial Meniscus** and 4. **Lateral Meniscus**: tear of that meniscus: any tear pattern (horizontal, oblique, radial, longitudinal, bucket-handle, flap, complex, root), signal reaching an articular surface (grade 3), displaced fragment, or maceration. Intrasubstance degeneration (grade 1 to 2) is `neg` p 0.10; extrusion alone is `uncertain` 0.40. Cues: menisco, menisküs, μηνίσκος, Meniskus, ménisque, menisk, meniscus, мениск; horn: cuerno, boynuz, κέρας, Hinterhorn/Vorderhorn, corne, rog, hoorn, рог.
5. **Medial OA**, 6. **Lateral OA**, 7. **PF OA**: osteoarthritis of that compartment (medial femorotibial, lateral femorotibial, patellofemoral). Positive: OA, arthrosis or gonarthrosis named for the compartment; high-grade (III to IV) or full-thickness cartilage loss; or osteophytes with cartilage thinning or joint-space narrowing. Chondromalacia or chondropathy grade I to II alone is `uncertain` 0.40. Compartment rules as above. Patella, trochlea and retropatellar cartilage belong to PF. Cues: artrosis, gonartrosis, condromalacia, osteofito; artroz, kondromalazi, osteofit; οστεοαρθρίτιδα, χονδρομαλακία, οστεόφυτα; Arthrose, Chondromalazie, Osteophyt; arthrose, chondropathie, ostéophyte; artroza, hondromalacija, osteofit; artrose, kraakbeenlijden, osteofyt; артроз, хондромаляция, остеофит.
8. **Effusion**: joint effusion of any size. "Physiological" or "small physiological amount of fluid" is `neg` p 0.10; minimal or trace effusion uses p 0.85. Cues: derrame, sıvı/efüzyon, υγρό/συλλογή, Erguss, épanchement, izljev, hydrops/vocht, выпот/жидкость.
9. **Synovitis**: synovitis, synovial thickening or proliferation, hypertrophic synovium, or Hoffa fat-pad edema (p 0.85). "Effusion-synovitis" sets both. Cues: sinovitis, sinovit, αρθρική υμενίτιδα, Synovitis/Synovialitis, synovite, sinovitis, synovitis, синовит.
10. **Baker's**: popliteal (Baker) cyst or distended semimembranosus–gastrocnemius bursa, including ruptured. Parameniscal and ganglion cysts do not count. Cues: quiste de Baker/poplíteo, Baker kisti, κύστη Baker, Bakerzyste, kyste poplité/de Baker, Bakerova cista, bakercyste, киста Бейкера.
11. **Contusion**: traumatic bone contusion, bone bruise, or bone marrow edema with a traumatic pattern (kissing, pivot-shift, impaction or post-traumatic context). Degenerative subchondral edema with OA, subchondral cysts, osteonecrosis or stress reaction is `neg` p 0.15. Bone marrow edema without a stated cause is `uncertain` 0.45. Cues: contusión, edema óseo; kontüzyon, kemik iliği ödemi; οστική θλάση, οίδημα μυελού; Knochenprellung, Knochenmarködem, bone bruise; contusion osseuse, œdème osseux; kontuzija, edem koštane srži; botkneuzing, beenmergoedeem; ушиб, отёк костного мозга.
12. **Fracture**: any current or recent fracture: osteochondral, impaction, avulsion (including Segond), plateau, patellar, or subchondral insufficiency fracture, or a trabecular fracture line. Bone bruise alone is `neg` (see above). An old healed fracture is `uncertain` 0.50. Cues: fractura, kırık, κάταγμα, Fraktur, fracture, prijelom/fraktura, fractuur, перелом.
