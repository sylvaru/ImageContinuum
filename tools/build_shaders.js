#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const entrypoints = [
  {
    name: "VSMain",
    dxSuffix: "vs",
    dxProfile: "vs_6_5",
    vkSuffix: "vert",
    vkProfile: "vs_6_6",
  },
  {
    name: "PSMain",
    dxSuffix: "ps",
    dxProfile: "ps_6_5",
    vkSuffix: "frag",
    vkProfile: "ps_6_6",
  },
  {
    name: "CSMain",
    dxSuffix: "cs",
    dxProfile: "cs_6_5",
    vkSuffix: "comp",
    vkProfile: "cs_6_6",
  },
];

function parseArgs(argv) {
  const options = {
    dxc: process.env.DXC || "dxc",
    debug: false,
  };

  for (let i = 0; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === "--debug") {
      options.debug = true;
    } else if (arg === "--dxc") {
      options.dxc = argv[++i];
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function walkShaders(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkShaders(fullPath, out);
    } else if (
      entry.isFile() &&
      entry.name.endsWith(".hlsl") &&
      !entry.name.startsWith("_")
    ) {
      out.push(fullPath);
    }
  }
  return out.sort();
}

function readWithIncludes(filePath, seen = new Set()) {
  const key = path.resolve(filePath);
  if (seen.has(key)) {
    return "";
  }
  seen.add(key);

  const source = fs.readFileSync(filePath, "utf8");
  const includePattern = /^\s*#include\s+"([^"]+)"/gm;
  let expanded = source;
  let match = null;
  while ((match = includePattern.exec(source)) !== null) {
    const includePath = path.resolve(path.dirname(filePath), match[1]);
    if (fs.existsSync(includePath)) {
      expanded += "\n" + readWithIncludes(includePath, seen);
    }
  }
  return expanded;
}

function hasEntry(source, entrypoint) {
  return new RegExp(`\\b${entrypoint}\\s*\\(`).test(source);
}

function run(command, args) {
  console.log([command, ...args].join(" "));
  const result = spawnSync(command, args, { stdio: "inherit", shell: false });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`${command} failed with exit code ${result.status}`);
  }
}

function compileShader(options, sourcePath, outputRoot) {
  const source = readWithIncludes(sourcePath);
  const stem = path.basename(sourcePath, ".hlsl");
  const commonFlags = options.debug ? ["-Zi", "-Qembed_debug"] : ["-O3"];
  const vulkanRegisterFlags = [
    "-fvk-b-shift",
    "0",
    "0",
    "-fvk-t-shift",
    "1",
    "0",
    "-fvk-s-shift",
    "100",
    "0",
  ];

  let compiled = 0;
  for (const entrypoint of entrypoints) {
    if (!hasEntry(source, entrypoint.name)) {
      continue;
    }

    const dxOut = path.join(
      outputRoot,
      "dx12",
      `${stem}.${entrypoint.dxSuffix}.dxil`
    );
    const vkOut = path.join(
      outputRoot,
      "vulkan",
      `${stem}.${entrypoint.vkSuffix}.spv`
    );

    run(options.dxc, [
      ...commonFlags,
      "-T",
      entrypoint.dxProfile,
      "-E",
      entrypoint.name,
      "-D",
      "IC_TARGET_DX12=1",
      "-Fo",
      dxOut,
      sourcePath,
    ]);

    run(options.dxc, [
      ...commonFlags,
      ...(entrypoint.dxSuffix === "cs" ? [] : vulkanRegisterFlags),
      "-spirv",
      "-fspv-target-env=vulkan1.3",
      "-T",
      entrypoint.vkProfile,
      "-E",
      entrypoint.name,
      "-D",
      "IC_TARGET_VULKAN=1",
      "-Fo",
      vkOut,
      sourcePath,
    ]);

    compiled += 2;
  }
  return compiled;
}

function main() {
  const root = path.resolve(__dirname, "..");
  const shaderRoot = path.join(root, "demo", "res", "shaders");
  const outputRoot = path.join(root, "demo", "res", "compiled_shaders");
  const options = parseArgs(process.argv.slice(2));

  fs.mkdirSync(path.join(outputRoot, "dx12"), { recursive: true });
  fs.mkdirSync(path.join(outputRoot, "vulkan"), { recursive: true });

  let total = 0;
  for (const shader of walkShaders(shaderRoot)) {
    total += compileShader(options, shader, outputRoot);
  }

  console.log(`Compiled ${total} shader artifacts.`);
}

try {
  main();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
