// electron-builder afterPack hook: stamps the ATM icon and version info onto the .exe
// with resedit (pure JS, so the installer can be built on Linux/macOS as well as Windows).
const fs = require("fs"), path = require("path");
module.exports = async function (context) {
  if (context.electronPlatformName !== "win32") return;
  const ResEdit = require("resedit");
  const exe = path.join(context.appOutDir, `${context.packager.appInfo.productFilename}.exe`);
  const data = fs.readFileSync(exe);
  const pe = ResEdit.NtExecutable.from(data, { ignoreCert: true });
  const res = ResEdit.NtExecutableResource.from(pe);
  const ico = ResEdit.Data.IconFile.from(fs.readFileSync(path.join(__dirname, "build", "icon.ico")));
  ResEdit.Resource.IconGroupEntry.replaceIconsForResource(res.entries, 1, 1033, ico.icons.map((i) => i.data));
  const [vi] = ResEdit.Resource.VersionInfo.fromEntries(res.entries);
  const v = context.packager.appInfo.version.split(".").map(Number);
  vi.setFileVersion(v[0], v[1], v[2], 0, 1033);
  vi.setProductVersion(v[0], v[1], v[2], 0, 1033);
  vi.setStringValues({ lang: 1033, codepage: 1200 }, {
    FileDescription: "ATM Stock Scan",
    ProductName: "ATM Stock Scan",
    CompanyName: "Absolute Tech Management LLP",
    LegalCopyright: "© 2026 Absolute Tech Management LLP",
    OriginalFilename: "ATM Stock Scan.exe",
    InternalName: "ATM Stock Scan",
  });
  vi.outputToResourceEntries(res.entries);
  res.outputResource(pe);
  fs.writeFileSync(exe, Buffer.from(pe.generate()));
  console.log("  • stamped icon + version info:", path.basename(exe));
};
