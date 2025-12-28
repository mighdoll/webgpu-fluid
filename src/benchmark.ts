import type { Accessor } from "solid-js";

interface Swappable {
  read: GPUTexture;
  write: GPUTexture;
  swap: () => void;
}

interface BindPair {
  read: () => GPUBindGroup;
}

const stats = (times: number[]) => {
  const sorted = [...times].sort((a, b) => a - b);
  const n = sorted.length;
  const q1 = sorted[Math.floor(n * 0.25)];
  const median = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];
  const iqr = q3 - q1;
  // Filter outliers (outside 1.5*IQR)
  const filtered = sorted.filter((t) => t >= q1 - 1.5 * iqr && t <= q3 + 1.5 * iqr);
  const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
  const stddev = Math.sqrt(filtered.reduce((sum, t) => sum + (t - mean) ** 2, 0) / filtered.length);
  return { median, mean, stddev, min: sorted[0], max: sorted[n - 1], q1, q3, filtered };
};

export const runBenchmark = async (
  device: GPUDevice,
  jacobiPipeline: GPURenderPipeline,
  jacobiComputePipeline: GPUComputePipeline,
  floatLayout: GPUBindGroupLayout,
  computeLayout: GPUBindGroupLayout,
  divergenceTex: Accessor<GPUTexture>,
  pressure: Swappable,
  pressurePair: BindPair,
  dwidth: () => number,
  dheight: () => number,
  colorAttachment: (view: GPUTexture) => GPURenderPassColorAttachment,
): Promise<void> => {
  const canTimestamp = device.features.has("timestamp-query");
  if (!canTimestamp) {
    console.warn("Timestamp queries not supported.");
    return;
  }

  const WARMUP_RUNS = 15;
  const MEASURE_RUNS = 50;
  const ITERATIONS_PER_RUN = 150;

  // Query set for per-iteration timing
  const querySetPerIter = device.createQuerySet({ type: "timestamp", count: ITERATIONS_PER_RUN * 2 });
  const resolveBufferPerIter = device.createBuffer({
    size: ITERATIONS_PER_RUN * 2 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const resultBufferPerIter = device.createBuffer({
    size: ITERATIONS_PER_RUN * 2 * 8,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  console.log("Starting Benchmark...");
  console.log(`Warmup: ${WARMUP_RUNS} runs, Measure: ${MEASURE_RUNS} runs × ${ITERATIONS_PER_RUN} iterations`);

  const divergenceReadGroup = device.createBindGroup({
    layout: floatLayout,
    label: "divergence read bind group",
    entries: [{ binding: 0, resource: divergenceTex().createView() }],
  });

  // Pre-create compute bind groups (alternating for ping-pong)
  const computeBindGroups = [
    device.createBindGroup({
      layout: computeLayout,
      entries: [
        { binding: 0, resource: divergenceTex().createView() },
        { binding: 1, resource: pressure.read.createView() },
        { binding: 2, resource: pressure.write.createView() },
      ],
    }),
    device.createBindGroup({
      layout: computeLayout,
      entries: [
        { binding: 0, resource: divergenceTex().createView() },
        { binding: 1, resource: pressure.write.createView() },
        { binding: 2, resource: pressure.read.createView() },
      ],
    }),
  ];

  const wgX = Math.ceil(dwidth() / 8);
  const wgY = Math.ceil(dheight() / 8);

  type TimingResult = { perIter: number; wall: number };
  type TimestampWrites = { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number };
  type WorkEncoder = (encoder: GPUCommandEncoder, iteration: number, timestamps?: TimestampWrites) => void;

  const measure = async (record: boolean, encodeWork: WorkEncoder): Promise<TimingResult> => {
    const commandEncoder = device.createCommandEncoder();
    for (let j = 0; j < ITERATIONS_PER_RUN; j++) {
      const timestamps = record
        ? { querySet: querySetPerIter, beginningOfPassWriteIndex: j * 2, endOfPassWriteIndex: j * 2 + 1 }
        : undefined;
      encodeWork(commandEncoder, j, timestamps);
    }
    device.queue.submit([commandEncoder.finish()]);

    if (!record) return { perIter: 0, wall: 0 };

    // Wait for GPU work to complete before resolving timestamps
    await device.queue.onSubmittedWorkDone();

    // Resolve timestamps in separate command buffer
    const resolveEncoder = device.createCommandEncoder();
    resolveEncoder.resolveQuerySet(querySetPerIter, 0, ITERATIONS_PER_RUN * 2, resolveBufferPerIter, 0);
    resolveEncoder.copyBufferToBuffer(resolveBufferPerIter, 0, resultBufferPerIter, 0, resultBufferPerIter.size);
    device.queue.submit([resolveEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await resultBufferPerIter.mapAsync(GPUMapMode.READ);
    const timesPerIter = new BigUint64Array(resultBufferPerIter.getMappedRange());
    let sumPerIter = 0;
    for (let j = 0; j < ITERATIONS_PER_RUN; j++) {
      sumPerIter += Number(timesPerIter[j * 2 + 1] - timesPerIter[j * 2]);
    }
    // Wall time = from start of first pass to end of last pass
    const wall = Number(timesPerIter[(ITERATIONS_PER_RUN - 1) * 2 + 1] - timesPerIter[0]);
    resultBufferPerIter.unmap();

    return { perIter: sumPerIter / ITERATIONS_PER_RUN, wall: wall / ITERATIONS_PER_RUN };
  };

  const encodeCompute: WorkEncoder = (encoder, j, timestamps) => {
    const pass = encoder.beginComputePass(timestamps ? { timestampWrites: timestamps } : undefined);
    pass.setPipeline(jacobiComputePipeline);
    pass.setBindGroup(0, computeBindGroups[j % 2]);
    pass.dispatchWorkgroups(wgX, wgY);
    pass.end();
  };

  const encodeFragment: WorkEncoder = (encoder, j, timestamps) => {
    const pass = encoder.beginRenderPass({
      colorAttachments: [colorAttachment(pressure.write)],
      ...(timestamps && { timestampWrites: timestamps }),
    });
    pass.setPipeline(jacobiPipeline);
    pass.setBindGroup(0, divergenceReadGroup);
    pass.setBindGroup(1, pressurePair.read());
    pass.draw(4, 1, 0, 0);
    pass.end();
    pressure.swap();
  };

  const measureCompute = (record: boolean) => measure(record, encodeCompute);
  const measureFragment = (record: boolean) => measure(record, encodeFragment);

  // Warmup - alternate to stabilize GPU clocks
  for (let i = 0; i < WARMUP_RUNS; i++) {
    await measureCompute(false);
    await measureFragment(false);
  }

  // Measure - interleaved to avoid thermal bias
  const computePerIter: number[] = [];
  const computeWall: number[] = [];
  const fragmentPerIter: number[] = [];
  const fragmentWall: number[] = [];

  for (let i = 0; i < MEASURE_RUNS; i++) {
    const c = await measureCompute(true);
    computePerIter.push(c.perIter);
    computeWall.push(c.wall);
    const f = await measureFragment(true);
    fragmentPerIter.push(f.perIter);
    fragmentWall.push(f.wall);
  }

  const computePerIterStats = stats(computePerIter);
  const computeWallStats = stats(computeWall);
  const fragmentPerIterStats = stats(fragmentPerIter);
  const fragmentWallStats = stats(fragmentWall);

  const fmt = (ns: number) => (ns / 1000).toFixed(2);

  console.log("=== Per-Iteration Timing (sum of individual pass durations) ===");
  console.log(`Fragment: median=${fmt(fragmentPerIterStats.median)} mean=${fmt(fragmentPerIterStats.mean)}±${fmt(fragmentPerIterStats.stddev)} us`);
  console.log(`Compute:  median=${fmt(computePerIterStats.median)} mean=${fmt(computePerIterStats.mean)}±${fmt(computePerIterStats.stddev)} us`);
  console.log(`Speedup: ${(fragmentPerIterStats.median / computePerIterStats.median).toFixed(2)}x`);

  console.log("=== Wall-Clock Timing (total time ÷ iterations) ===");
  console.log(`Fragment: median=${fmt(fragmentWallStats.median)} mean=${fmt(fragmentWallStats.mean)}±${fmt(fragmentWallStats.stddev)} us`);
  console.log(`Compute:  median=${fmt(computeWallStats.median)} mean=${fmt(computeWallStats.mean)}±${fmt(computeWallStats.stddev)} us`);
  console.log(`Speedup: ${(fragmentWallStats.median / computeWallStats.median).toFixed(2)}x`);

  console.log("=== Overhead (wall - perIter) ===");
  const computeOverhead = computeWallStats.median - computePerIterStats.median;
  const fragmentOverhead = fragmentWallStats.median - fragmentPerIterStats.median;
  console.log(`Fragment overhead: ${fmt(fragmentOverhead)} us/iter`);
  console.log(`Compute overhead:  ${fmt(computeOverhead)} us/iter`);

  // Cleanup
  querySetPerIter.destroy();
  resolveBufferPerIter.destroy();
  resultBufferPerIter.destroy();
};
