# Cough Detector Cross-Device Validation & Stability Audit Protocol

## Objective
Identify and validate the cough event detector and threshold that are **most stable across recording hardware** (Audio-Recorder, Phone A, Phone B, Phone C) **before** extracting audio embeddings (OPERA, CLAP) or training TB classifiers.

---

## 1. Strict Guidelines & Non-Negotiables
1. **Never use TB disease labels to tune the detector**:
   - Cough event detection is an acoustic event segmentation task, not a disease classifier. Using TB status to choose detection thresholds introduces severe label leakage and overfitting.
2. **Signal Properties & Listening-Based Validation Only**:
   - Validation relies on signal consistency, cross-device event count concordance, and auditory verification of detected sound intervals.
3. **No Synthetic or Fabricated Real-Data Results**:
   - Real audio resides on Kaggle (55 GB). The audit scripts run directly on Kaggle with real data. A `--test-synthetic` flag is available strictly for offline code/unit-test verification.

---

## 2. Cohort & Device Setup
- **Sample**: 20 patients deterministically sampled from the Zambia cohort with `has_ABC == True` and recordings present across all 4 devices (`random_seed = 42`).
- **Facilities Covered**: Kanyama (`Kan`, 7 patients), Chawama (`Cha`, 8 patients), Chainda-South (`Chai`, 5 patients).
- **Synchronous Devices per Patient**:
  1. `Audio-Recorder` (Dedicated linear reference audio recorder, 44.1 kHz)
  2. `Phone A` (Smartphone microphone with onboard AGC)
  3. `Phone B` (Smartphone microphone with onboard AGC)
  4. `Phone C` (Smartphone microphone with onboard AGC)

The complete list of 20 patients and their 80 audio files is recorded in [`sampled_patients_manifest.csv`](sampled_patients_manifest.csv).

---

## 3. Compared Detectors & Mathematical Formulations

### Signal Preprocessing (Uniform across Detectors)
1. Resample to $16\text{ kHz}$ mono float32.
2. 4th-order Butterworth highpass filter ($f_c = 80\text{ Hz}$) to eliminate microphone handling thumps, breathing wind noise, and DC offset.
3. Frame-level RMS energy envelope ($W = 50\text{ ms}$, step $H = 10\text{ ms}$):
   $$E[m] = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} x[m \cdot H + i]^2}$$
4. Moving average smoothing across 3 frames ($30\text{ ms}$).

### Detector 1: Median + $k \times \text{MAD}$ Detector
- **Baseline**:
  $$\text{med} = \text{median}(E)$$
  $$\text{MAD} = \text{median}(|E - \text{med}|)$$
- **Adaptive Threshold**:
  $$\text{Threshold} = \text{med} + k \times \text{MAD}$$
- **Evaluated Multipliers**: $k \in \{4.0, 5.0, 6.0, 7.0, 8.0\}$, with **$k = 6.0$ as PRIMARY**.
- **Scale-Invariance Property**:
  If a smartphone applies an analog gain or multiplier $\alpha$ to the input signal ($x' = \alpha x$), the envelope scales as $E' = \alpha E$. Because both median and MAD scale linearly ($\text{med}(\alpha E) = \alpha \text{med}(E)$, $\text{MAD}(\alpha E) = \alpha \text{MAD}(E)$), the normalized ratio $\frac{E'}{\text{Threshold}'} = \frac{E}{\text{Threshold}}$ is **strictly invariant to microphone gain and volume differences**.

### Detector 2: Relative dB Detector
- Decibel envelope:
  $$E_{\text{dB}}[m] = 20 \log_{10}(E[m] + 10^{-7})$$
- **Noise Floor Relative**:
  $$\text{NoiseFloor}_{\text{dB}} = \text{percentile}_{10}(E_{\text{dB}})$$
  $$\text{Threshold}_{\text{dB}} = \text{NoiseFloor}_{\text{dB}} + \Delta_{\text{dB}}, \quad \Delta_{\text{dB}} \in \{12, 15, 20, 25\}\text{ dB}$$
- **Peak Relative**:
  $$\text{Peak}_{\text{dB}} = \max(E_{\text{dB}})$$
  $$\text{Threshold}_{\text{dB}} = \text{Peak}_{\text{dB}} - \Delta_{\text{drop\_dB}}, \quad \Delta_{\text{drop\_dB}} \in \{15, 20, 25\}\text{ dB}$$

### Event Segmentation & Cough Constraints
- **Debouncing**: Contiguous active frames are merged if separated by $\le 150\text{ ms}$ (to unite the explosive phase and voiced/intermediate phase of a single cough).
- **Duration Gating**: Only intervals with duration $80\text{ ms} \le \text{dur} \le 1200\text{ ms}$ are accepted as candidate cough events.

---

## 4. Stability Metric: Cross-Device Coefficient of Variation (CV)
For patient $p$ coughing in the recording room, the true number of physical coughs is fixed.
For each detector, we obtain:
$$N_{\text{Rec}}, N_{\text{Phone A}}, N_{\text{Phone B}}, N_{\text{Phone C}}$$
- Patient Mean: $\mu_p = \frac{1}{4} \sum_d N_d$
- Patient Std: $\sigma_p = \sqrt{\frac{1}{3} \sum_d (N_d - \mu_p)^2}$
- Patient Cross-Device CV:
  $$\text{CV}_p = \frac{\sigma_p}{\mu_p} \quad (\text{if } \mu_p > 0 \text{ else } 0)$$
- **Cohort Mean CV**: $\overline{\text{CV}} = \frac{1}{20} \sum_{p=1}^{20} \text{CV}_p$ (Lower is better; 0 indicates identical event counts across all 4 devices).

---

## 5. Execution on Kaggle
1. Open a Kaggle notebook with dataset `googlehealthai/google-health-ai` attached.
2. Copy and paste [`scripts/kaggle_cough_detector_audit.py`](../../scripts/kaggle_cough_detector_audit.py) into a single code cell.
3. Run the cell. The audit runs across the 80 audio files in ~1-2 minutes.
4. Outputs are saved to `/kaggle/working/results/cough_detector_audit/` and zipped as `results_cough_detector_audit.zip`.
5. Download `results_cough_detector_audit.zip` and extract it into `results/cough_detector_audit/` in your local workspace.

