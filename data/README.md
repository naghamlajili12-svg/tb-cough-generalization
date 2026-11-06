# Data

**No raw data is stored in this repository.** All three datasets are (semi-) public and
must be obtained from their original providers under their respective terms.
Place downloaded data under `data/raw/<dataset>/` and the pipeline will populate
`data/interim/` (segmented coughs, validated labels) and `data/processed/`
(features, embeddings, metadata tables).

| Dataset | Source | Notes |
|---|---|---|
| CODA (multi-country) | CODA TB DREAM Challenge release ([data](https://www.synapse.org/Synapse:syn31472953/wiki/617828), [publication](https://www.nature.com/articles/s41597-024-03972-z))| Only half of the full dataset is public; we use Vietnam, Madagascar, Tanzania (single-device countries), solicited coughs only. |
| TBscreen (Kenya) | TBscreen study release ([data](https://zenodo.org/records/10431329), [publication](https://www.science.org/doi/10.1126/sciadv.adi0282)) | Passive vs forced cough kept separate; passive used for training, forced for validation only. |
| Zambia CIDRZ (HeAR) | CIDRZ / HeAR benchmark release ([data](https://www.kaggle.com/datasets/googlehealthai/google-health-ai), [publication](https://arxiv.org/abs/2403.02522)) | Sessions released as full recordings; Chainda-South site dropped; audio recorder used for validation only. |


### Layout
```
data/
├── raw/                       # downloaded datasets (git-ignored)
├── interim/                   # segmented coughs, relabeled subjects (git-ignored)
└── processed/                 # features, embeddings, metadata tables (git-ignored)
└── zambia_change_log.csv      # log of the changes of subject label for Zambia CIDRZ dataset
```
See the paper's "Study-specific dataset subsets" table for the exact subjects,
cough counts, and TB-positive rates per subgroup that the pipeline reproduces.

### Re-labeling of Zambia CIDRZ dataset

Screening was done to filter for audio samples that are wrongly labeled within the dataset. Specifically, the paticipant ID was wrongly entered for the audio recording. By manually listening to the recording, one can recover the correct paticipant ID. (In the recording, the staff would first announce the paticipant ID, then the paticipant would start coughing.)
Screening was done is two ways automatically:
- Calculate the maximum cross-correlation between recordings from the same participant, but from different devices. Check for low values to flag for possible mis-alignment of participant label and recording.
- Check for participant IDs that are not registered in the meta data file.

A csv file records all re-labeling done (so far). It is possible that still some mis-labeling exists. As part of the steps to process the data, we access information in the CSV file and make the relabeling.
