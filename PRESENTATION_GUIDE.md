# ThreatNet presentation flow — fictional Riya case

All names, numbers, registrations, source records, CCTV frames, and the workbook
in this demo are fictional. The app presents investigative leads for human review,
not findings of identity or guilt.

## Before presenting

Run `start-demo.ps1 -ResetEnvironment` from PowerShell after Python has been
installed. Open `http://127.0.0.1:3000` and select **CASE-RIYA-001**.

## 3–5 minute walkthrough

1. **Command view (30 seconds)** — Introduce Riya Sharma as a fictional student.
   Point out the case-theory evidence and the four ranked review signals. Explain
   that the payment, contact pattern, vehicle sighting, and route surveillance are
   source-linked leads, not a conclusion by themselves.

2. **Timeline (60 seconds)** — Follow Riya's routine, then show the three vehicle
   sightings of Rohan's sedan, the repeated calls, the Rohan–Vikram contact, the
   fictional payment record, and the Riverside incident window. Open the alert and
   explain that it flags a conflict between Rohan's stated account and a vehicle
   sighting for investigator review.

3. **Mind map (60 seconds)** — Highlight the links between Riya, Rohan, Vikram,
   their fictional vehicles, the college/home/party locations, and Riverside
   service road. Hover an evidence node to show its source notes.

4. **CCTV intake (45 seconds)** — Choose “Synthetic Riverside CAM-07 vehicle
   sighting.” Show the local sampled frames and explain that they are fictional
   reviewable frames. Do not claim automatic plate or identity confirmation.

5. **Suspect screening (45 seconds)** — Upload
   `backend/storage/demo-assets/case-riya-suspect-screening.xlsx`. Show that Rohan
   and Vikram are ranked because the worksheet deliberately supplies higher route
   and call-link signals. Explain that the spreadsheet only prioritizes review.

6. **Close (20 seconds)** — Summarize the investigative theory: the combined
   fictional records support review of Rohan as a possible organizer and Vikram as
   a possible direct perpetrator. Human verification, source authentication, and
   legal process would be required before any real-world action.
