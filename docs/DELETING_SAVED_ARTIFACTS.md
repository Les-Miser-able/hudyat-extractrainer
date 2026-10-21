# Delete saved models and prepared datasets

In the native app:

- Choose a model in **Model runs**, then **Delete selected model…** in the inspector.
- Choose a prepared dataset in **Run configuration**, then **Delete selected dataset…**.

In the browser dashboard, use **Delete model** in the model comparison actions or
**Delete dataset** beside the prepared dataset in the source library.

The confirmation names the exact selected item. Deletion is disabled while a job
or camera prediction is active and checked again by the backend. Source recordings
and other saved datasets/models remain available. Deleting a dataset keeps trained
models, although benchmarking or continuing a run that needs the original dataset
requires restoring it or selecting an available compatible dataset. For older
category-level datasets, deletion moves only their known generated artifacts and
keeps named datasets beneath the same category folder.

Deleted files move to `.trash/<timestamp>-<id>/` inside the project. This removes
them from dataset/model selectors without permanently erasing them. Each deletion
includes `deletion.json`, which lists original and stored paths. To recover an item,
stop current jobs, move the listed files back to their original paths without
overwriting newer files, then refresh the app. Trash is ignored by Git and continues
to occupy disk space until you remove it yourself.

A deleted model's manual review is archived as `model_record.json` in its trash
folder and removed from active reviews, so a reused run name starts fresh. Existing
prediction tests keep their frozen model descriptions and measurements. Existing
study reports and exported ZIPs are saved snapshots; rebuild reports after deletion
to update the study comparison. Exported copies are not deleted with the original.
