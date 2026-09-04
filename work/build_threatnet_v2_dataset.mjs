import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = path.resolve("outputs", "threatnet-v2-input-dataset");
const outputPath = path.join(outputDir, "ThreatNet-V2-Input-Dataset.xlsx");
const previewDir = path.join(outputDir, "previews");
const workbook = Workbook.create();

const readme = workbook.worksheets.add("Read me");
const suspects = workbook.worksheets.add("Suspects");
const narratives = workbook.worksheets.add("Narrative Text");
const events = workbook.worksheets.add("Event Input");

const colors = {
  navy: "#173653",
  blue: "#245B80",
  paleBlue: "#EAF4FA",
  amber: "#FFF2CC",
  border: "#B9C9D6",
  text: "#17212B",
  muted: "#54636F",
};

function setTitle(sheet, title, subtitle) {
  sheet.getRange("A1:F1").merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format = {
    font: { name: "Arial", size: 16, bold: true, color: colors.text },
    horizontalAlignment: "left",
  };
  sheet.getRange("A2:F2").merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2").format = {
    font: { name: "Arial", size: 10, italic: true, color: colors.muted },
    wrapText: true,
  };
}

function formatHeader(range) {
  range.format = {
    fill: colors.navy,
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
}

function formatData(range) {
  range.format = {
    font: { name: "Arial", size: 10, color: colors.text },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: colors.border },
  };
}

readme.showGridLines = false;
setTitle(
  readme,
  "ThreatNet V2 Synthetic Input Dataset",
  "Synthetic training/demo data only. Similarity scores, extracted relationships, and alerts require human verification."
);
readme.getRange("A4:B9").values = [
  ["Input type", "Where to enter it"],
  ["Suspect screening workbook", "ThreatNet web app > Suspect screening > choose this workbook."],
  ["Narrative text", "FastAPI docs > POST /api/case/{case_id}/intelligence/extract > copy the Raw Text value."],
  ["Timeline event", "ThreatNet web app > Timeline > Add timeline event."],
  ["CCTV and reference portrait", "Run python seed.py. It creates a synthetic clip and portrait under backend/storage/cctv/v2-demo/."],
  ["Face matching", "FastAPI docs > POST /api/cctv/match-face. Use the synthetic reference portrait after running seed.py."],
];
formatHeader(readme.getRange("A4:B4"));
formatData(readme.getRange("A5:B9"));
readme.getRange("A4:B9").format.borders = { preset: "all", style: "thin", color: colors.border };
readme.getRange("A4").format.columnWidth = 27;
readme.getRange("B4").format.columnWidth = 83;
readme.getRange("A4:B9").format.autofitRows();

suspects.showGridLines = false;
suspects.getRange("A1:F1").values = [[
  "Suspect ID",
  "Name",
  "Area",
  "Vehicle",
  "Route Match",
  "Call Link Count",
]];
suspects.getRange("A2:F7").values = [
  ["DEMO-S-001", "Arjun Rao", "Jubilee Hills", "White hatchback", "High", 5],
  ["DEMO-S-002", "Priya Nair", "Banjara Hills", "Silver sedan", "Medium", 3],
  ["DEMO-S-003", "Rohan Das", "Madhapur", "Motorcycle", "Low", 1],
  ["DEMO-S-004", "Meera Iyer", "Panjagutta", "Black SUV", "Medium", 6],
  ["DEMO-S-005", "Kabir Sen", "Jubilee Hills", "Delivery van", "High", 2],
  ["DEMO-S-006", "Anika Bose", "Madhapur", "White scooter", "Low", 0],
];
formatHeader(suspects.getRange("A1:F1"));
formatData(suspects.getRange("A2:F7"));
suspects.getRange("A1:F7").format.borders = { preset: "all", style: "thin", color: colors.border };
suspects.getRange("F2:F7").format.numberFormat = "0";
suspects.getRange("A1").format.columnWidth = 16;
suspects.getRange("B1").format.columnWidth = 20;
suspects.getRange("C1").format.columnWidth = 18;
suspects.getRange("D1").format.columnWidth = 21;
suspects.getRange("E1").format.columnWidth = 16;
suspects.getRange("F1").format.columnWidth = 17;
suspects.getRange("A1:F7").format.autofitRows();
suspects.freezePanes.freezeRows(1);
suspects.tables.add("A1:F7", true, "SuspectScreeningData");

narratives.showGridLines = false;
setTitle(
  narratives,
  "Narrative Extraction Input",
  "Copy the raw text into the intelligence extraction API. The sample exercises SIGHTED_AT, CALLED, and OWNS relationships."
);
narratives.getRange("A4:E4").values = [[
  "Source",
  "Source Type",
  "Default Time",
  "Default Location",
  "Raw Text",
]];
narratives.getRange("A5:E6").values = [
  [
    "Synthetic witness note 01",
    "statement",
    "2026-09-04T20:00:00",
    "Banjara Hills",
    "Arjun Rao stated he was at Banjara Hills at 2026-09-04T20:00:00. Arjun Rao called Priya Nair at 2026-09-04T20:01:00 from Banjara Hills. Arjun Rao owns a white hatchback.",
  ],
  [
    "Synthetic witness note 02",
    "statement",
    "2026-09-04T20:05:00",
    "Jubilee Hills",
    "Meera Iyer was seen at Jubilee Hills at 2026-09-04T20:05:00. Meera Iyer called Kabir Sen at 2026-09-04T20:06:00 from Jubilee Hills.",
  ],
];
formatHeader(narratives.getRange("A4:E4"));
formatData(narratives.getRange("A5:E6"));
narratives.getRange("A4:E6").format.borders = { preset: "all", style: "thin", color: colors.border };
narratives.getRange("A5:D6").format.fill = colors.amber;
narratives.getRange("A4").format.columnWidth = 24;
narratives.getRange("B4").format.columnWidth = 16;
narratives.getRange("C4").format.columnWidth = 19;
narratives.getRange("D4").format.columnWidth = 20;
narratives.getRange("E4").format.columnWidth = 90;
narratives.getRange("A4:E6").format.autofitRows();
narratives.freezePanes.freezeRows(4);

events.showGridLines = false;
setTitle(
  events,
  "Timeline Event Input",
  "Add these one at a time in the Timeline view. Entity ID and coordinates are available for the API when a linked presence event is needed."
);
events.getRange("A4:F4").values = [[
  "Event Label",
  "Location",
  "ISO Timestamp",
  "Entity Name",
  "Kind",
  "Coordinates",
]];
events.getRange("A5:F6").values = [
  [
    "Statemented presence: Arjun Rao",
    "Banjara Hills",
    "2026-09-04T20:00:00",
    "Arjun Rao",
    "presence",
    "17.4126, 78.4482",
  ],
  [
    "Camera review queued",
    "Jubilee Hills",
    "2026-09-04T20:03:00",
    "",
    "observation",
    "17.4326, 78.4070",
  ],
];
formatHeader(events.getRange("A4:F4"));
formatData(events.getRange("A5:F6"));
events.getRange("A4:F6").format.borders = { preset: "all", style: "thin", color: colors.border };
events.getRange("A5:F6").format.fill = colors.amber;
events.getRange("A4").format.columnWidth = 34;
events.getRange("B4").format.columnWidth = 20;
events.getRange("C4").format.columnWidth = 19;
events.getRange("D4").format.columnWidth = 20;
events.getRange("E4").format.columnWidth = 15;
events.getRange("F4").format.columnWidth = 22;
events.getRange("A4:F6").format.autofitRows();
events.freezePanes.freezeRows(4);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Read me", "Suspects", "Narrative Text", "Event Input"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1.5, format: "png" });
  await fs.writeFile(path.join(previewDir, sheetName.replaceAll(" ", "-").toLowerCase() + ".png"), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 6000,
  tableMaxRows: 10,
  tableMaxCols: 8,
});
await fs.writeFile(path.join(outputDir, "workbook-inspection.ndjson"), inspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
await fs.writeFile(path.join(outputDir, "formula-errors.ndjson"), errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
