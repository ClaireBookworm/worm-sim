# C. elegans Biophysical Simulation

Closed-loop: Jaxley biophysical RNN → muscle readout → MuJoCo worm.

## Quick Rundown

1. **Collect** – Run Jaxley + sine-wave muscles, record (voltages, muscles).
2. **Train** – Fit linear readout: voltages → 96 muscles (ridge regression, optional wiring mask).
3. **Run** – Closed-loop sim: sensory state → Jaxley → readout → muscles → physics.

## Commands

```bash
# 1. Collect training data
python -m biophysical_integration.collect_muscle_training_data \
  --checkpoint models/polyak_corr/input_trainable_voltage_biophys \
  --net-cache models/polyak_corr/input_trainable_voltage_biophys/network_synapse_location_always_soma.pkl \
  --output-dir biophysical_integration/data \
  --n-steps 2000

# 2. Train muscle readout
python biophysical_integration/muscle_readout.py \
  --voltages biophysical_integration/data/voltages.npy \
  --muscles biophysical_integration/data/muscles.npy \
  --save biophysical_integration/checkpoints/muscle_readout.npz \
  --ridge 0.1 \
  --wiring-mask BAAIWorm/eworm_learn/components/param/connection/neuron_muscle.xlsx

# 3. Run closed-loop simulation
mjpython run_biophysical_sim.py \
  --checkpoint models/polyak_corr/input_trainable_voltage_biophys \
  --net-cache models/polyak_corr/input_trainable_voltage_biophys/network_synapse_location_always_soma.pkl \
  --checkpoint-epoch 5000 \
  --readout biophysical_integration/checkpoints/muscle_readout.npz \
  --n-steps 2000 \
  --verbose \
  --save-output test_run2/ \
  --save-video test_run2/output.mp4 \
  --render
```

(Omit `--wiring-mask` if the Excel file is missing. Use `python` instead of `mjpython` if not rendering.)
