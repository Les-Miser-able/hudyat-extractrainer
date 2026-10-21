# HUDYAT Methods placeholder guide

The draft in `hudyat-methods-draft.txt` describes the saved prototype runs underlying the current manuscript's approximately 98.25% category-mean results. It does not claim that the revised Da Best protocol has been carried out.

| Placeholder | What is needed | Where to obtain it |
|---|---|---|
| INTERVIEW_DATE | Interview date, or month and year if the exact date is unavailable | Visit records, messages, or interview notes. The user confirmed an interview, with teaching ongoing during the visit. |
| CONTACT_ROLE | Sir Genesis's official role at the center | Confirm with the team or school. The user identified Sir Genesis as the primary contact; his official designation is not yet established. |
| INTERVIEW_FINDINGS | Difficulties, teaching practices, and requested support actually discussed, with links to HUDYAT features | Interview notes or the team's recollection. The user has not yet supplied the interview's specific findings. |
| DATASET_REFERENCE | Dataset name, author, version, URL, citation, and permitted use | Original download page, dataset README, or download history. A source filename is not enough to establish authorship or permission. |
| SIGNER_PROFILE | Who the three signers were by role, FSL experience, selection, and researcher/participant status | Ask the recording team. Use anonymized descriptions in the paper. |
| RECORDING_SETUP | Recording dates, device, frame size/FPS, distance, light/background, and repetitions | Original video metadata plus the recording team's recollection or session notes. Metadata alone cannot establish distance or lighting. |
| SIGN_VALIDATION | FSL references and any actual review of labels and demonstrations | Reference videos/materials and reviewer records. Do not claim expert validation if none occurred. |
| CONSENT_AND_PERMISSIONS | Consent for recordings, school access, recording storage, and any required ethics review | Actual consent/permission forms or adviser/school records. Do not invent an approval or identifier. |
| MODEL_SELECTION | How the final four prototype runs were selected | Model records and experiment notes. State any use of test results or informal live performance rather than claiming validation-only selection. |
| BENCHMARK_DEFINITION | Adopted thresholds and when they were established | Research protocol and benchmark configuration. Existing project targets are 85% for category classification metrics and 70% minimum per-sign recall; confirm their adoption and timing scope. |
| LIVE_TEST_PROTOCOL | Actual test status, signers/overlap, trials, trial order, settings, timeouts, error scoring, and timestamps | Trial logs and an executed test plan. If not done, retain planned status until completion. The proposed 780 attempts are not completed evidence. |
| TEST_ENVIRONMENT | Device hardware, OS/browser, backend, camera settings, lighting, and distance | Test-day records and browser/device reports, not estimates from another device. |
| DEPLOYMENT_DETAILS | Model releases, application build, and parity checks actually completed | Export manifests, model catalog, build identity, and parity reports. |
| FUNCTIONAL_TEST_PROCEDURE | Test status, tester, environment, cases, and expected/actual/pass-fail records | A case log or executed checklist. The proposed 40 checks are not completed evidence. |

Start the discussion with requirements gathering, the dataset source, and signer/recording details. Technical settings and source counts are already supported by the saved configurations and manifests. If fresh models are trained under the revised protocol, replace the old seed, augmentation, normalization, class-weighting, counts, model-selection, and evaluation details as one consistent revision.
