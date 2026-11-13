# Research Notes

Notes and TODOs moved out of the original `main.ipynb` (cells 145–157), kept verbatim. Now at `notebooks/archive/main.ipynb`.

# TO DO: 



###### Use Grad-CAM or heatmap visualization to interpret model focus.

###### Implement data augmentation early to reduce overfitting.

###### Downsample images to 224×224 or 256×256 to save resources without performance loss.



###### Use sklearn train test split to split the data into train and validation set

###### Also address the class imbalance

###### More EDA on the Data---graphs and tables to show the data stuff and training decision made based on that

###### Image Augmentation to increase the data samples



###### Train the model again after completing all of the above steps for up to 100 epochs
###### In the mean time also look for more powerful computer to train the model on


###### Heatmap after the model has been trained.....to see where the model mostly localizes its predication ....and comapre it with the initial heatmap of the label bbox data


###### PA and AP view also plays a huge role in how the model will train.....(`FUTURE CONSIDERATION`: we can pass the data into the model giving it context of the AP and PA view so it has this `view` context aware)

---

Take the model we have trained now(the best one).....run inference on it and get the return bbox data from it....and plot the heatmap...the heatmap of our trained model (map50 = 0.25) on the final prediciton of the model

For post model evaluation heatmap

Only add to the heatmap when the confiednce score in greater then a ceratin threshold
if confidence_score > threshold:
* .....
* .....
* heatmap[y_min:y_max, x_min:x_max] += 1

Also dont add 1 but rather add the confidence score
if confidence_score > threshold:
* .....
* .....
* heatmap[y_min:y_max, x_min:x_max] += confidence_score


Why we add a confidence score......

If you just add +1 for every predicted box above the threshold, all boxes are treated equally — even if one is 51% confident and another is 99%. But intuitively:

A 99% confidence box should contribute more to the heatmap.

A 51% confidence box might be a borderline false positive — it should contribute less.

By adding the confidence score:

Areas where the model consistently predicts with high confidence will "light up" more.

You get a more nuanced and informative visualization.

---

### Here are some tips this time around to improve the result with the model
1. Verify Label Quality
* Visualize random samples with bounding boxes.
* Check for tight boxes, accurate positioning, and no false positives.
* Ensure multi-lesion images have multiple boxes.

2. Balance Your Dataset
* Check for class imbalance (e.g., many normal X-rays vs few pneumonia).
* Consider oversampling pneumonia cases.
* Optionally, downsample normal cases or balance batches.

3. Use Image-Level Metadata
* Use DICOM metadata (e.g., ViewPosition, PatientAge) to filter or group data.
* Prefer consistent views (e.g., only PA or AP) to avoid confusing patterns.

4. Data Augmentation
* Use or customize YOLO26’s built-in augmentations.
* Apply flips, scaling, blur, brightness, and contrast adjustments.
* More aggressive augmentation can help detect small or subtle patterns.

5. Try Larger YOLO26 Models
* Instead of yolo26n, try yolo26s, yolo26m, or yolo26l depending on GPU capacity.
* Larger models capture more features and improve accuracy.

6. Use Mosaic Augmentation
* Enabled by default in YOLO26.
* Especially effective for small and sparse objects like pneumonia patches.
* Make sure it's not accidentally turned off.

7. Train Longer & Use Cosine Learning Rate
* Train for 100+ epochs if loss hasn’t plateaued.
* Use a cosine LR schedule for smoother convergence.
* Example: scheduler='cosine', lr0=0.001, lrf=0.01.

8. Evaluate on a Clean Validation/Test Set
* Use a separate, representative validation/test split.
* Run model.val() or model.predict() to evaluate performance.
* Avoid leakage between training and test sets.

9. Filter Out Bad or Corrupt Images
* Check for black images, invalid DICOMs, or low-contrast junk data.
* Remove lateral views or mislabeled cases, if any.

10. (Optional) Handle Grayscale Input Smartly
* X-rays are grayscale, but YOLO expects 3-channel RGB.
* Convert grayscale to 3-channel by duplicating channels.
* Advanced: Modify model input layer to accept 1-channel images (not necessary, but possible).

---

```python
### FOR FUTURE: If u wanna expand the dataset to finetune the model (more dataset to finetune the model)

# Scour internet to possibly find more pneumonia data that u can use to train your model....
# Following are some of the publicly available dataset that contain pnuemonia related data.......future consideration to combine them with the RSNA Pnuemonia dataset.....




# NIH ChestX-ray14
# Large public dataset with multiple disease labels including pneumonia. Mostly image-level labels but some bounding boxes available.

# CheXpert (Stanford)
# Very large, labeled chest X-ray dataset. Has uncertain labels, so careful handling needed.

# MIMIC-CXR
# Huge dataset with chest X-rays and rich metadata from ICU patients. Requires credentialed access.

# COVID-19 X-ray datasets
# Include pneumonia cases (viral pneumonia). Useful for broader pneumonia modeling.

# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
```

---

🔬 4. Use Synthetic Data (Optional Advanced)
If your model still struggles with generalization, consider:

* GANs to generate pneumonia-like lungs (requires extra work)

* Copy-Paste augmentation: Cut pneumonia areas from positive images and paste them on negatives (experimental, works better with masks)

-----------------------------------------------


🧪 3. (Advanced) Synthetic Data
* Generating pneumonia-like images using GANs or Copy-Paste is powerful, but:

    * It’s research-level

    * Needs masks or well-localized regions

    * Takes time and experimentation

✔️ High risk–high reward
✘ Might not be necessary for RSNA

---

it does matter what kind of augmentation you apply, especially for object detection on a medical dataset like RSNA Pneumonia Detection.

Here's a breakdown:

✅ Augmentations That Help:
These simulate realistic variations in X-rays without breaking the medical meaning:

Horizontal flip (A.HorizontalFlip)
✅ Safe and common. Lungs are symmetric; flipping doesn’t usually distort meaning.

Small rotations / shifts / scales (A.ShiftScaleRotate)
✅ Slight changes are okay — simulate different patient positioning.
⚠️ Avoid large shifts/rotations that could move pneumonia out of frame.

Brightness/contrast changes (A.RandomBrightnessContrast)
✅ Good for robustness to exposure variation across machines.

Gaussian noise or blur (light)
✅ Adds robustness to imaging artifacts, but use lightly.

⚠️ Augmentations That Can Hurt:
Vertical flip
🚫 Not realistic — lungs don’t appear upside down in valid chest X-rays.

Cutout / GridDropout
⚠️ Risky unless used carefully; might obscure critical pneumonia areas.

Heavy rotation (>15°), scale, crop
🚫 Can make the bounding boxes invalid or cut out the pneumonia region.

Color jitter / hue / saturation
🚫 Medical images are grayscale — color transforms don’t apply.

---
