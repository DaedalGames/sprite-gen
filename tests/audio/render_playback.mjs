// SPDX-License-Identifier: Apache-2.0
// node tests/audio/render_playback.mjs <cases.json> <output-dir>
// PLAYWRIGHT_MODULE may name an installed Playwright entry point.
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
const { chromium, webkit } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const [caseFile, outputDir] = process.argv.slice(2);
const cases = JSON.parse(await readFile(caseFile, 'utf8'));
await mkdir(outputDir, { recursive: true });
const modulePath = resolve('sprite_gen/audio/sprite-gen-audio-v2.mjs');
const moduleBytes = await readFile(modulePath);
const digest = createHash('sha256').update(moduleBytes).digest('hex');
const reports = [];
for (const [name, engine] of Object.entries({ chromium, webkit })) {
  const browser = await engine.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.goto('about:blank');
    for (const rate of [44100, 48000]) {
      const synthetic = await page.evaluate(async ({ code, rate }) => {
        const { prepareAudioBuffer, createAudioSource } = await import(code);
        const check = (ok, message) => { if (!ok) throw new Error(message); };
        // Non-zero origin and a frame count whose independent rounding differs.
        const meta = { version: 2, sample_rate: 48000, start_frame: 13,
          end_frame: 144014, overlap_frames: 12001 };
        const first = Math.round(meta.start_frame * rate / 48000);
        const n = Math.round((meta.end_frame-meta.start_frame)*rate/48000);
        const length = Math.round((meta.end_frame-meta.start_frame-meta.overlap_frames)*rate/48000);
        const count = n-length;
        const input = new AudioBuffer({ sampleRate: rate, length: first+n, numberOfChannels: 2 });
        for (let c=0;c<2;c++) {
          const x=input.getChannelData(c);
          for (let i=0;i<x.length;i++) x[i]=.2*Math.cos(2*Math.PI*(c ? 73 : 1000)*i/rate)+.1*i/x.length;
        }
        const before = input.getChannelData(0).slice();
        const out = prepareAudioBuffer(input, meta);
        check(out.length === length, 'one-frame rounding drift');
        check(prepareAudioBuffer(input, meta) === out, 'decoded input was prepared twice');
        check(prepareAudioBuffer(out, meta) === out, 'prepared output was crossfaded twice');
        check(input.getChannelData(0).every((v,i)=>v===before[i]), 'decoded source mutated');
        for (let c=0;c<2;c++) {
          const x=input.getChannelData(c), y=out.getChannelData(c);
          const expected=x[first+count]-x[first+count-1];
          check((y[0]-y.at(-1))-expected === 0, 'added wrap error');
          const middle=n-2*count;
          check((y[middle]-y[middle-1])-(x[first+n-count]-x[first+n-count-1])===0, 'added join error');
          // The detector must reject an artificial discontinuity, even though
          // a high-frequency signal's normal adjacent difference is allowed.
          const control=y.slice(); control[0]+=.125;
          check(Math.abs((control[0]-control.at(-1))-expected)>.12, 'broken negative control');
        }
        let rejected=false;
        try { prepareAudioBuffer(out, { ...meta, overlap_frames: 12002 }); } catch { rejected=true; }
        check(rejected, 'changed metadata reused prepared audio');
        const ctx=new OfflineAudioContext(2, length*3, rate);
        const source=createAudioSource(ctx,input,{loop:true,playback:meta});
        check(source.buffer===out && source.loop && source.loopStart===0 && source.loopEnd===out.duration, 'source bypasses preparation');
        source.connect(ctx.destination); source.start();
        const rendered=await ctx.startRendering();
        let maxError=0;
        for(let c=0;c<2;c++) for(let i=0;i<rendered.length;i++) {
          maxError=Math.max(maxError,Math.abs(rendered.getChannelData(c)[i]-out.getChannelData(c)[i%length]));
        }
        check(maxError<1e-7, 'rendered looping differs from prepared buffer');
        return { length, count, added_wrap_error:0, added_join_error:0, control_detected:true, render_error:maxError };
      }, { code: `data:text/javascript;base64,${moduleBytes.toString('base64')}`, rate });
      reports.push({ browser:name, browser_version:browser.version(), rate, synthetic });
      for (const item of cases) {
        const manifestPath = resolve(dirname(caseFile), item.manifest);
        const pack = JSON.parse(await readFile(manifestPath,'utf8'));
        const record = item.id ? pack.audio[item.id] : pack;
        const shipped = record.player ? await readFile(resolve(dirname(manifestPath), record.player.path)) : moduleBytes;
        assert.equal(createHash('sha256').update(shipped).digest('hex'), digest);
        if (record.player) assert.equal(record.player.sha256, digest);
        for (const fmt of ['ogg','m4a']) {
          const bytes = await readFile(resolve(dirname(manifestPath),record.files[fmt].path));
          assert.equal(createHash('sha256').update(bytes).digest('hex'),record.files[fmt].sha256);
          const result = await page.evaluate(async ({code,encoded,record,rate}) => {
            const { prepareAudioBuffer, createAudioSource } = await import(code);
            const data=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
            const decoder=new OfflineAudioContext(2,1,rate);
            const decoded=await decoder.decodeAudioData(data.buffer);
            const buffer=prepareAudioBuffer(decoded,record.playback);
            const m=record.playback, scale=decoded.sampleRate/m.sample_rate;
            const first=Math.round(m.start_frame*scale), n=Math.round((m.end_frame-m.start_frame)*scale);
            const length=Math.round((m.end_frame-m.start_frame-m.overlap_frames)*scale), count=n-length;
            if (buffer.length!==length) throw new Error('wrong prepared duration');
            const context=new OfflineAudioContext(buffer.numberOfChannels,length*3,rate);
            const source=createAudioSource(context,decoded,record);
            source.connect(context.destination); source.start();
            const rendered=await context.startRendering();
            let wrap=0, join=0, renderError=0, controls=0;
            for (let c=0;c<buffer.numberOfChannels;c++) {
              const x=decoded.getChannelData(c), y=buffer.getChannelData(c), r=rendered.getChannelData(c);
              if (record.loop) {
                const baseline=x[first+count]-x[first+count-1];
                for(let lap=1;lap<3;lap++) {
                  const at=lap*length;
                  wrap=Math.max(wrap,Math.abs((r[at]-r[at-1])-baseline));
                  if(Math.abs((r[at]+.125-r[at-1])-baseline)>.12) controls++;
                }
                const at=n-2*count;
                join=Math.max(join,Math.abs((y[at]-y[at-1])-(x[first+n-count]-x[first+n-count-1])));
              }
              for(let i=0;i<r.length;i++) {
                const expected=record.loop ? y[i%length] : (i<length ? y[i] : 0);
                renderError=Math.max(renderError,Math.abs(r[i]-expected));
              }
            }
            const interleaved=new Float32Array(rendered.length*rendered.numberOfChannels);
            for(let i=0;i<rendered.length;i++) for(let c=0;c<rendered.numberOfChannels;c++)
              interleaved[i*rendered.numberOfChannels+c]=rendered.getChannelData(c)[i];
            const u8=new Uint8Array(interleaved.buffer);
            let binary=''; for(let i=0;i<u8.length;i+=16384) binary+=String.fromCharCode(...u8.subarray(i,i+16384));
            return { decoded_frames:decoded.length, length, overlap:count, channels:rendered.numberOfChannels,
              added_wrap_error:wrap, added_join_error:join, render_error:renderError, controls_detected:controls,
              pcm:btoa(binary) };
          },{code:`data:text/javascript;base64,${shipped.toString('base64')}`,encoded:bytes.toString('base64'),record,rate});
          assert.equal(result.added_wrap_error,0);
          assert.equal(result.added_join_error,0);
          assert.ok(result.render_error<1e-7);
          if(record.loop) assert.equal(result.controls_detected,result.channels*2);
          const pcmName=`${item.name}-${name}-${rate}-${fmt}.f32`;
          await writeFile(resolve(outputDir,pcmName),Buffer.from(result.pcm,'base64'));
          delete result.pcm;
          reports.push({ name:item.name,browser:name,rate,fmt,target_lufs:record.target_lufs,pcm:pcmName,...result });
          await writeFile(resolve(outputDir,'render.json'),JSON.stringify({module_sha256:digest,reports},null,2));
        }
      }
    }
  } finally { await browser.close(); }
}
await writeFile(resolve(outputDir,'render.json'),JSON.stringify({module_sha256:digest,reports},null,2));
console.log(JSON.stringify({passed:reports.length,module_sha256:digest}));
