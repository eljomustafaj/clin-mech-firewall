import { copyFile } from "node:fs/promises";

for (const asset of ["index.html", "style.css"]) {
  await copyFile(new URL(asset, import.meta.url), new URL(`dist/${asset}`, import.meta.url));
}
