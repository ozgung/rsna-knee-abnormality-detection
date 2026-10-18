# OAI semi-quantitative MRI labels for the 12 RSNA targets (2026-10-12)

Documentation-only review of public pages; no NDA login, download or terms acceptance. Bracketed keys map to the source list at the end. "Unverified" means no fetched source states it.

## 1. OAI reading datasets

| Dataset | Reader / vendor | Knees | Visits | Notes |
|---|---|---|---|---|
| `kMRI_SQ_MOAKS_BICLxx` (+ `kMRI_FNIH_SQ_MOAKS_BICLxx`), Project 22 | BICL (Boston); two radiologists, initials FWR and AG [A][B] | 600 (FNIH nested case-control: 194 cases, 406 controls, matched on KL and BMI) | Baseline (V00), 12 and 24 mo | Released 2015 [A][B] |
| Further baseline MOAKS: projects 22, 30, 46, 47, 63A/B/D/E/F, 65 | Unverified (several enriched or case-control substudies) | A 2026 paper matched 2,899 knees from 2,392 participants [C] | Baseline | Not in the 2016 overview [A]; check the current NDA data dictionary |
| `kMRI_SQ_BICLxx`, `kMRI_SQ_BLKSBML_BICLxx` (WORMS and BLOKS, Project 10) | Boston Univ. (Guermazi) [A] | 115, high cartilage-loss risk [D] | Baseline, 24 mo | Cartilage, meniscus, BML only [D]; no ligament or cyst items seen |
| `kXR_SQ_BUxx` (KL, JSN, osteophytes; Project 15, 37/42) | Boston Univ. (Felson) [A] | 4,492 participants (8,984 knees) | V00 to 96 mo | Other radiographic features scored only if KL >= 2 [A] |

**Plausible 2,399-knee source (unverified inference):** the pooled baseline MOAKS projects above (2,392 participants, 2,899 knees [C]) are the same order of magnitude. FNIH alone (600) is too small. No source ties Raptor to either number. The 2016 overview reports no per-subcohort counts.

## 2. Target mapping

MOAKS facts are from [E] and [F] (the MOST protocol, not OAI); the OAI field list is unverified. Check `kMRI_SQ_MOAKS_BICL00` variable guide for each field before relying on it.

| Target | Candidate field and semantics | Proposed binarization | Confidence |
|---|---|---|---|
| ACL tear | MOAKS ACL: normal vs complete tear; reconstruction flag [E]. MOST protocol also lists ACL/PCL tears [F]. Mucoid degeneration was read separately [G]. | Tear = 1; exclude reconstructed knees | Approximate. Complete tears are rare (12 complete, 25 partial in 600 [H]); partial-tear coding unverified |
| MCL tear | Not a MOAKS item [E]. Coronal IW TSE could show MCL [I] | None | Unsupported |
| Medial / lateral meniscus tear | MOAKS morphology per anterior horn, body, posterior horn: vertical, horizontal, radial, complex, root tear, maceration; signal-only is not a tear [E] | Any tear or maceration in that meniscus; extrusion alone is not a tear | Good. Degenerative tears dominate |
| Medial / lateral OA | Cartilage lesion size and full-thickness loss, osteophytes 0-3, BML, meniscal damage [E]. Published MRI-OA Delphi rule: osteophyte or full-thickness loss plus supporting features [J][K]. Radiographic fallback: KL and medial/lateral JSN [A] | Compartment-level Delphi-style rule, or full-thickness loss plus osteophyte | Approximate. The competition's OA definition is unverified; KL is a whole-knee radiographic grade; lateral OA is rare in an OA cohort (unverified) |
| PF OA | MOAKS patella and trochlea cartilage and osteophytes [E]. Baseline radiographs are PA fixed-flexion, so there is no PF view [I] | PF Delphi-style rule | Approximate to weak; MOAKS only |
| Effusion | Effusion-synovitis 0-3 (small, medium, large) [E][B] | >= 2 (>= 1 sensitivity variant) | Good |
| Synovitis | Hoffa-synovitis 0-3 on sagittal FS images [E][B]; OAI is non-contrast | >= 2, optionally OR effusion-synovitis 3 | Approximate; no contrast-based synovitis |
| Baker's cyst | MOAKS popliteal cyst present/absent [E]; scored in OAI by one radiologist [L] (26-28% prevalence in 340 KL 0 knees) | Present = 1 | Good if the field exists in the public file (unverified) |
| Contusion | BML size per subregion 0-3 (<33%, 33-66%, >66%) [E] | None by default; BML >= 2 is only a candidate proxy | **Not accepted as a Contusion label** without separate validation. OAI BMLs are mostly degenerative, not traumatic, and the positive rate would be inflated relative to bone bruise. Mask Contusion unless validated. |
| Fracture | No MOAKS fracture item [E] | None (mask the loss) | Unsupported |

Population mismatch: OAI is ages 45-79, with or at risk of symptomatic knee OA [I], and Siemens 3 T, four US sites. The competition is referral MRI from 16 sites on five continents [M]. Expect degenerative-tear and OA-heavy label priors. Use OAI for masked, target-specific supervision, not for prevalence calibration.

## 3. Sequences and domain shift

OAI acquired bilateral knees on 3 T Siemens Trio [I]. Parameters are from the 2006 protocol Appendix F [I].

| Series | Key parameters | Competition analogue |
|---|---|---|
| Sag IW TSE FS | TR 3200, TE 30, 3 mm, 37 slices, FOV 160 | Sag PD FS (best match for BML, effusion, cyst, ACL) |
| Cor IW TSE (no FS) | TR 3850, TE 29, 3 mm, 41 slices | Cor PD (menisci, MCL/LCL) |
| Sag 3D DESS WE | TR 16.3, TE 4.7, 0.7 mm, 160 slices; coronal and axial MPR reformats | No native match; axial and coronal MPR are not PD FS |
| Cor 3D FLASH WE, sag T2 map (7 echoes, TE 10-70) | Right knee only | T2 map is non-FS; no T2 FS equivalent |

Table 4.2 of the protocol appears to label the coronal IW TSE as "FS", but Appendix F and the FNIH reading paper list it as non-FS [I][B]; Appendix F is used here. The FNIH MOAKS reads used coronal IW TSE, sagittal DESS and sagittal IW TSE FS [B]. Implications (competition series as stated in the task brief, unverified): no native axial PD FS or T2 FS; one vendor and one protocol; 3 mm slices with no gap on TSE. Train on the three readable series with strong contrast and resolution augmentation, and add axial PD FS via DESS MPR only as an ablation.

## 4. Join keys

- Participant ID (`ID`), visit prefix (`V00` baseline, `V01` 12 mo; others per `VisitPrefixDefinitions.pdf` [A]), `SIDE` (1 = right, 2 = left), `READPRJ` (project). Duplicate (ID, SIDE) rows across projects need a precedence rule [A][C].
- Images join through the barcode: DICOM AccessionNumber = `VxxMRBARCD` in the `MRIxx` metadata datasets [A].
- NDA says its subject IDs differ from the OAI IDs used in the ASCII/SAS downloads and raw images; the mapping table is stated to be on the page but was not visible to me. That it is a GUID is unverified [N].
- Keep all visits and both knees of one participant in one fold.

## 5. Download size

No documented size for the MRI subset. The whole archive is about 26.6 M images [O]. NDA states a 20 TB per user per 30-day cap, and says prepackaged sets are uncompressed and cannot be split by scan type, so use the query tool to filter [N]. My arithmetic, uncompressed: DESS about 47 MB (384x384x160, 16-bit); each TSE series about 10-15 MB; about 70 MB per knee for the three core series, so 2,400-2,900 knees are about 0.17-0.2 TB, plus MPR series if requested. Unverified.

## Sources

[A] https://nda.nih.gov/static/docs/ImageAssessmentDataOverview.pdf
[B] https://pmc.ncbi.nlm.nih.gov/articles/PMC5105263/
[C] https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1908547/full
[D] https://research.manchester.ac.uk/en/publications/comparison-of-bloks-and-worms-scoring-systems-part-i-cross-sectio/
[E] https://radiopaedia.org/articles/mri-osteoarthritis-knee-score-moaks
[F] https://agingresearchbiobank.nia.nih.gov/studies/most/documents/download/Dataset_Descriptions/DatasetDescriptionReadingProtocol_V79MOAKS.pdf/
[G] https://healthimaging.com/node/88776
[H] https://cris.fau.de/publications/120891804
[I] https://nda.nih.gov/static/docs/StudyDesignProtocolAndAppendices.pdf
[J] https://pmc.ncbi.nlm.nih.gov/articles/PMC12796920
[K] doi:10.1016/j.joca.2011.04.017 (Hunter 2011 Delphi MRI-OA definition, cited in [J])
[L] https://pmc.ncbi.nlm.nih.gov/articles/PMC6202768
[M] https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge
[N] https://nda.nih.gov/general-query.html?q=query%3Dfeatured-datasets%3AOsteoarthritis+Initiative+%28OAI%29
[O] https://www.re3data.org/repository/r3d100011843

**Qualifications (16:05 UTC, after Codex review).** Every field marked unverified (ACL/PCL, popliteal cyst, and anything outside FNIH project 22) stays **masked** until it is confirmed in the actual variable guide. Degenerative BML is not an accepted Contusion label without separate validation. Mappings with similar names are not treated as equivalent by default. This document carries no download or terms action, and OAI remains deferred.
