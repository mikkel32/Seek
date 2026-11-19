// Integer inference for the exported search-trace model. History is oldest first;
// each completed depth supplies a Uint8Array in legal-move order, seven bytes per move.
function tracePointers(weights, history, count) {
    const width = 16, vocabulary = 68, fields = 7;
    if (weights.length !== 3*vocabulary*width + 3*width || count < 1 ||
        history.length < 1 || history.some(frame => frame.length !== count*fields))
        throw new Error("Invalid trace dimensions");
    const bias = 3*vocabulary*width, heads = bias+width, result = [];
    for (let s=0; s<count; s++) {
        const hidden = Int32Array.from(weights.subarray(bias, heads));
        for (let lag=0; lag<Math.min(3,history.length); lag++) {
            const frame = history[history.length-1-lag];
            for (let field=0; field<fields; field++) {
                const token=frame[s*fields+field];
                if (token >= vocabulary) throw new Error("Invalid trace token");
                const offset=(lag*vocabulary+token)*width;
                for (let unit=0; unit<width; unit++) hidden[unit]+=weights[offset+unit];
            }
        }
        let next=0, challenger=0;
        for (let unit=0; unit<width; unit++) {
            const activation=Math.min(384,Math.max(0,hidden[unit]));
            next+=activation*weights[heads+2*unit];
            challenger+=activation*weights[heads+2*unit+1];
        }
        result.push([next,challenger]);
    }
    return result;
}

function traceProbabilities(logits, head, excluded=-1) {
    const maximum=Math.max(...logits.map((row,i) => i===excluded ? -Infinity : row[head]));
    const mass=logits.map((row,i) => i===excluded ? 0 : Math.exp((row[head]-maximum)/32768));
    const sum=mass.reduce((a,b) => a+b,0);
    return mass.map(v => sum ? v/sum : 0);
}
if (typeof module !== "undefined") module.exports = { tracePointers, traceProbabilities };
