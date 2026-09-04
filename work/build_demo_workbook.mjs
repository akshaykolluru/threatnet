import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = new URL("../outputs/", import.meta.url).pathname;
await fs.mkdir(outputDir, { recursive: true });
const workbook = Workbook.create();
const suspects = workbook.worksheets.add("Suspects");
const calls = workbook.worksheets.add("Call Logs");
const guide = workbook.worksheets.add("Narrowing Guide");
const suspectHeaders = [["Suspect ID", "Name", "Age", "Area", "Vehicle", "Phone", "Last Seen", "Route Match", "Call Link Count", "Review Score", "Review Status"]];
const suspectRows = Array.from({ length: 500 }, (_, index) => {
  const n = index + 1;
  const route = n % 7 === 0 ? "High" : n % 3 === 0 ? "Medium" : "Low";
  const linkedCalls = n % 11 === 0 ? 8 : n % 5 === 0 ? 3 : n % 2;
  const score = Math.min(99, (route === "High" ? 48 : route === "Medium" ? 25 : 8) + linkedCalls * 5 + (n % 9));
  return [`S-${String(n).padStart(4, "0")}`, `Synthetic Person ${n}`, 20 + (n % 42), n % 4 === 0 ? "Jubilee Hills" : n % 4 === 1 ? "Banjara Hills" : n % 4 === 2 ? "Madhapur" : "Somajiguda", n % 6 === 0 ? "White hatchback" : "Unlinked", `+91-90000-${String(n).padStart(5, "0")}`, `2026-09-${String(4 + (n % 3)).padStart(2, "0")} ${String(19 + (n % 3)).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`, route, linkedCalls, score / 100, score >= 60 ? "Shortlist" : "Review later"];
});
suspects.getRange("A1:K501").values = suspectHeaders.concat(suspectRows);
calls.getRange("A1:H501").values = [["Call ID", "From Phone", "To Phone", "Date", "Time", "Duration Sec", "Tower Area", "Case Link"], ...Array.from({ length: 500 }, (_, index) => { const n = index + 1; return [`CALL-${String(n).padStart(4, "0")}`, `+91-90000-${String(n).padStart(5, "0")}`, `+91-98888-${String((n * 7) % 500).padStart(5, "0")}`, `2026-09-${String(4 + (n % 3)).padStart(2, "0")}`, `${String(19 + (n % 3)).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`, 30 + (n * 13) % 600, n % 2 ? "Jubilee Hills" : "Banjara Hills", "CASE-101"]; })];
guide.getRange("A1:B8").values = [["Narrowing Guide", "Synthetic demo input for CASE-101"], ["Rows", 500], ["Output", "Ranked shortlist for investigator review"], ["Signal 1", "Route Match: High > Medium > Low"], ["Signal 2", "Call Link Count increases review priority"], ["Signal 3", "Time and area are supporting context only"], ["Guardrail", "A score is not proof, guilt, or an identification"], ["Source", "Synthetic data generated for the SIH prototype"]];
for (const sheet of [suspects, calls, guide]) { sheet.showGridLines = false; const used = sheet.getUsedRange(); used.format.font = { name: "Arial", size: 10, color: "#1f2937" }; used.format.autofitColumns(); used.format.autofitRows(); sheet.getRange("A1:Z1").format = { fill: "#173b72", font: { bold: true, color: "#ffffff" }, verticalAlignment: "center" }; sheet.freezePanes.freezeRows(1); }
suspects.getRange("J2:J501").format.numberFormat = [["0.0%"]];
const output = await SpreadsheetFile.exportXlsx(workbook); await output.save(`${outputDir}/sih-investigation-demo.xlsx`);
const preview = await workbook.render({ sheetName: "Suspects", range: "A1:K16", scale: 1 }); await fs.writeFile(`${outputDir}/sih-investigation-demo-preview.png`, new Uint8Array(await preview.arrayBuffer()));
