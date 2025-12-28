import benchFragWGSL from "../shaders/bench_frag.wesl?static";
import benchComputeWGSL from "../shaders/bench_compute.wesl?static";
import vertWGSL from "../shaders/vert.wesl?static";

const stats = (times: number[]) => {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = sorted[Math.floor(n * 0.25)];
  const median = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  const filtered = sorted.filter((t) => t >= q1 - 1.5 * iqr && t <= q3 + 1.5 * iqr);
  const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  const stddev = Math.sqrt(filtered.reduce((sum, t) => sum + (t - mean) ** 2, 0) / filtered.length);
  return { median, mean, stddev, min: sorted[0], max: sorted[n - 1] };
};

export const runBenchmark = async (device: GPUDevice, width: number, height: number): Promise<void> => {
  if (!device.features.has("timestamp-query")) {
    console.warn("Timestamp queries not supported.");
    return;
  }

  const WARMUP_RUNS = 5;
  const MEASURE_RUNS = 20;
  const ITERATIONS_PER_RUN = 50;

  // Three textures: divergence (input), pressure (input), new_pressure (output)
  const divergenceTex = device.createTexture({
    size: [width, height],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const pressureTex = device.createTexture({
    size: [width, height],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const outputTex = device.createTexture({
    size: [width, height],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  // Fragment pipeline - using layout: "auto" (will have TWO bind groups)
  const fragModule = device.createShaderModule({ code: benchFragWGSL });
  const vertModule = device.createShaderModule({ code: vertWGSL });
  const fragPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: vertModule, entryPoint: "vert" },
    fragment: {
      module: fragModule,
      entryPoint: "main",
      targets: [{ format: "r32float" }],
    },
  });

  // Compute pipeline - using layout: "auto" (will have ONE bind group with 3 bindings)
  const computeModule = device.createShaderModule({ code: benchComputeWGSL });
  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: computeModule, entryPoint: "main" },
  });

  // Fragment bind groups: TWO groups (group 0: divergence, group 1: pressure)
  const fragBindGroup0 = device.createBindGroup({
    layout: fragPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: divergenceTex.createView() }],
  });
  const fragBindGroup1 = device.createBindGroup({
    layout: fragPipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: pressureTex.createView() }],
  });

  // Compute bind group: ONE group with 3 bindings
  const computeBind = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: divergenceTex.createView() },
      { binding: 1, resource: pressureTex.createView() },
      { binding: 2, resource: outputTex.createView() },
    ],
  });

  // Timing resources
  const querySet = device.createQuerySet({ type: "timestamp", count: ITERATIONS_PER_RUN * 2 });
  const resolveBuffer = device.createBuffer({
    size: ITERATIONS_PER_RUN * 2 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const resultBuffer = device.createBuffer({
    size: ITERATIONS_PER_RUN * 2 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  console.log("Starting Benchmark...");
  console.log(`Warmup: ${WARMUP_RUNS} runs, Measure: ${MEASURE_RUNS} runs × ${ITERATIONS_PER_RUN} iterations`);

  const wgX = Math.ceil(width / 8);
  const wgY = Math.ceil(height / 8);

  type TimingResult = { perIter: number; wall: number };

  const measure = async (
    record: boolean,
    encodeWork: (enc: GPUCommandEncoder, i: number, ts?: { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number }) => void,
  ): Promise<TimingResult> => {
    const encoder = device.createCommandEncoder();
    for (let j = 0; j < ITERATIONS_PER_RUN; j++) {
      const ts = record ? { querySet, beginningOfPassWriteIndex: j * 2, endOfPassWriteIndex: j * 2 + 1 } : undefined;
      encodeWork(encoder, j, ts);
    }
    device.queue.submit([encoder.finish()]);

    if (!record) return { perIter: 0, wall: 0 };

    await device.queue.onSubmittedWorkDone();
    const resolveEnc = device.createCommandEncoder();
    resolveEnc.resolveQuerySet(querySet, 0, ITERATIONS_PER_RUN * 2, resolveBuffer, 0);
    resolveEnc.copyBufferToBuffer(resolveBuffer, 0, resultBuffer, 0, resultBuffer.size);
    device.queue.submit([resolveEnc.finish()]);
    await device.queue.onSubmittedWorkDone();

    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(resultBuffer.getMappedRange());
    let sum = 0;
    for (let j = 0; j < ITERATIONS_PER_RUN; j++) {
      sum += Number(times[j * 2 + 1] - times[j * 2]);
    }
    const wall = Number(times[(ITERATIONS_PER_RUN - 1) * 2 + 1] - times[0]);
    resultBuffer.unmap();
    return { perIter: sum / ITERATIONS_PER_RUN, wall: wall / ITERATIONS_PER_RUN };
  };

  type TS = { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number };

  const encodeCompute = (enc: GPUCommandEncoder, _: number, ts?: TS) => {
    const pass = enc.beginComputePass(ts ? { timestampWrites: ts } : undefined);
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, computeBind);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  };

  const encodeFragment = (enc: GPUCommandEncoder, _: number, ts?: TS) => {
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: outputTex.createView(), loadOp: "load", storeOp: "store" }],
      ...(ts && { timestampWrites: ts }),
    });
    pass.setPipeline(fragPipeline);
    pass.setBindGroup(0, fragBindGroup0);  // divergence
    pass.setBindGroup(1, fragBindGroup1);  // pressure
    pass.draw(4);
    pass.end();
  };

  // Warmup
  for (let i = 0; i < WARMUP_RUNS; i++) {
    await measure(false, encodeCompute);
    await measure(false, encodeFragment);
  }

  // Measure
  const computePerIter: number[] = [];
  const computeWall: number[] = [];
  const fragmentPerIter: number[] = [];
  const fragmentWall: number[] = [];

  for (let i = 0; i < MEASURE_RUNS; i++) {
    const c = await measure(true, encodeCompute);
    computePerIter.push(c.perIter);
    computeWall.push(c.wall);
    const f = await measure(true, encodeFragment);
    fragmentPerIter.push(f.perIter);
    fragmentWall.push(f.wall);
  }

  const cpi = stats(computePerIter), cw = stats(computeWall);
  const fpi = stats(fragmentPerIter), fw = stats(fragmentWall);
  const fmt = (ns: number) => (ns / 1000).toFixed(2);

  console.log("=== Per-Iteration Timing (sum of individual pass durations) ===");
  console.log(`Fragment: median=${fmt(fpi.median)} mean=${fmt(fpi.mean)}±${fmt(fpi.stddev)} us`);
  console.log(`Compute:  median=${fmt(cpi.median)} mean=${fmt(cpi.mean)}±${fmt(cpi.stddev)} us`);
  console.log(`Speedup: ${(fpi.median / cpi.median).toFixed(2)}x`);

  console.log("=== Wall-Clock Timing (total time ÷ iterations) ===");
  console.log(`Fragment: median=${fmt(fw.median)} mean=${fmt(fw.mean)}±${fmt(fw.stddev)} us`);
  console.log(`Compute:  median=${fmt(cw.median)} mean=${fmt(cw.mean)}±${fmt(cw.stddev)} us`);
  console.log(`Speedup: ${(fw.median / cw.median).toFixed(2)}x`);

  console.log("=== Overhead (wall - perIter) ===");
  console.log(`Fragment overhead: ${fmt(fw.median - fpi.median)} us/iter`);
  console.log(`Compute overhead:  ${fmt(cw.median - cpi.median)} us/iter`);

  // Cleanup
  querySet.destroy();
  resolveBuffer.destroy();
  resultBuffer.destroy();
  divergenceTex.destroy();
  pressureTex.destroy();
  outputTex.destroy();
};
