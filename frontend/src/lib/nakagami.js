/**
 * Theoretical average BER for BPSK over Nakagami-m fading.
 *
 * Closed-form expression (Simon & Alouini, eq. 8.108):
 *   BER(γ̄) = (1/2)(1 - μ · Σ_{k=0}^{m-1} C(2k,k) · ((1-μ²)/4)^k)
 *   where μ = sqrt(γ̄ / (γ̄ + m)), γ̄ = average SINR per symbol
 *
 * For m=1 (Rayleigh), this reduces to: BER = 0.5 · (1 - sqrt(γ̄/(γ̄+1)))
 */

function binom2kk(k) {
  // C(2k, k) = (2k)! / (k!)^2  — precomputed iteratively
  if (k === 0) return 1;
  let val = 1;
  for (let i = 1; i <= k; i++) {
    val = (val * (2 * i - 1) * 2) / (i * 2);
  }
  // Direct formula: product of (2i-1)/i for i=1..k, times 2^0 ... cleaner:
  let num = 1, den = 1;
  for (let i = 0; i < k; i++) {
    num *= (2 * k - i);
    den *= (i + 1);
  }
  return num / den;
}

export function nakagamiBerCurve(m = 1, snrMinDb = -5, snrMaxDb = 30, points = 70) {
  const result = [];
  const step = (snrMaxDb - snrMinDb) / points;
  const mInt = Math.round(m);  // m must be positive integer for closed form

  for (let snrDb = snrMinDb; snrDb <= snrMaxDb; snrDb += step) {
    const gammBar = Math.pow(10, snrDb / 10);
    const mu = Math.sqrt(gammBar / (gammBar + mInt));

    let sum = 0;
    for (let k = 0; k < mInt; k++) {
      sum += binom2kk(k) * Math.pow((1 - mu * mu) / 4, k);
    }
    const ber = Math.max(0.5 * (1 - mu * sum), 1e-7);
    result.push({ x: parseFloat(snrDb.toFixed(2)), y: ber });
  }
  return result;
}
