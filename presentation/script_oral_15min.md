# Oral script - XAI FER presentation (15 minutes)

Speaking goal: do not read the slides. The slides give the structure; the speech explains why each choice was made and what each result proves.

## Slide 1 - Title (30 s)

Hello, today we present our XAI final project: *Evaluating and Improving Explainability in Facial Emotion Recognition*.

The goal is to study the explainability of a facial emotion recognition model. We use RAF-DB, a CNN trained from scratch, and we compare three explanation methods: Grad-CAM, LIME and SHAP.

The key point is not only to generate heatmaps, but to test whether these explanations are actually consistent with the model's decisions.

## Slide 2 - Outline (35 s)

The presentation follows four parts.

First, the problem: why explainability matters in facial emotion recognition. Second, the method: dataset, model and XAI pipeline. Third, the results: model performance, visual examples, quantitative XAI metrics and masking analysis. Finally, we discuss limitations and take-home messages.

## Slide 3 - Motivation (1 min 10)

Facial Emotion Recognition consists in predicting an emotion from a face image. It can be useful in human-computer interaction, education or healthcare, but it is also sensitive: a prediction may influence a human interpretation or a decision.

The issue is that CNNs can be predictive without being transparent. A correct prediction can still be made for the wrong reason: the model may rely on the background, lighting, framing, or artifacts.

Our research question is therefore: are explanations produced by Grad-CAM, LIME and SHAP truly faithful to model decisions in FER?

## Slide 4 - Related work (1 min)

Our project is positioned between two groups of work.

On one side, FER datasets and models, especially RAF-DB, which provides real-world face images with annotated expressions. On the other side, XAI methods: Grad-CAM for gradient-based heatmaps, LIME for local perturbation-based explanations, and SHAP for attribution inspired by Shapley values.

However, work such as Adebayo et al. and Yeh et al. shows that a saliency map may look convincing without being faithful. This is exactly what we want to test: not only visualize, but evaluate.

## Slide 5 - Data and protocol (1 min)

We use RAF-DB with 7 emotions: neutral, happy, sad, surprise, fear, disgust and anger.

Images are preprocessed as 128 by 128 grayscale images. We use fixed splits: around 25,000 training images, 5,383 validation images and 5,384 test images.

Reproducibility is important: everything is configured through YAML files, with a fixed global seed. The final model is trained for 150 epochs, and the best checkpoint comes from epoch 135.

One important point: our goal is not to beat the FER state of the art, but to obtain a realistic model whose explanations can be analyzed.

## Slide 6 - Model (1 min 15)

The model is a CNN trained from scratch, without transfer learning. This keeps the setting controlled: we analyze our own model, not a large pretrained backbone with external biases.

The architecture has 4 convolutional blocks with 32, 64, 128 and 256 channels. We use depthwise separable convolutions to reduce complexity, residual connections, and SE attention in deeper blocks.

Training uses AdamW, cosine annealing, dropout, and several augmentations: horizontal flip, rotation, translation, brightness, contrast and random erasing. These choices aim to make the model robust without changing the core XAI question.

## Slide 7 - XAI pipeline (1 min 15)

We compare three methods.

Grad-CAM uses gradients of the target class in the last convolutional layer. It gives a smooth map that is easy to read, but sometimes coarse.

LIME masks image regions and learns locally which regions influence the prediction. It is closer to a local causal test, but strongly depends on the segmentation.

SHAP approximates region contributions through coalitions. It is conceptually strong, but expensive, and in our implementation it relies on a region grid.

Essential rule: for a given image, all three methods explain the same target, namely the class predicted by the model.

## Slide 8 - Evaluating explanations (1 min 20)

We do not evaluate explanations only visually. We use three families of tests.

First, faithfulness. With deletion, we progressively remove the regions considered important: if the explanation is good, confidence should drop quickly. With insertion, we start from a degraded image and progressively add important regions: if the explanation is good, confidence should recover quickly.

Second, robustness: we apply small perturbations such as noise, brightness changes or translation. A stable explanation should not change completely.

Third, facial masking: we mask the eyes, the mouth or the background, then observe confidence drop and prediction changes.

## Slide 9 - Model performance (1 min 20)

On the test set, the model reaches 55.11 percent accuracy, 54.65 percent balanced accuracy, and 49.11 percent macro-F1.

This is not a perfect model, but it is useful for our objective: it has classes that are relatively well learned, such as happy and surprise, and difficult classes, such as fear.

The confusion matrix shows natural confusions: fear is often confused with sad, surprise or anger; sad and neutral are also sometimes confused. These errors are interesting for XAI analysis, because they let us check whether the model fails for visually plausible reasons.

## Slide 10 - Correct case: happy -> happy (1 min 15)

Here, the true label is happy and the model predicts happy with high confidence.

Grad-CAM gives a smooth map that mainly covers the face and the mouth. LIME and SHAP produce more discrete regions. The interpretation is coherent: for the emotion happy, the mouth and smile-related facial traits are important cues.

This slide shows that explanations can be plausible when the model is correct. But the real question is: what happens when the model is wrong?

## Slide 11 - Failure case: anger -> happy (1 min 30)

This image is central to the presentation.

The true label is anger, but the model predicts happy. Now, the three methods correctly explain the same target class: happy.

What we observe is interesting: the methods highlight regions around the eyes and especially around the mouth. Visually, there is an apparent smile, or at least a facial configuration that can mislead the model.

So the error is not random. The explanation reveals a semantic confusion: the model associates some local cues with happy even though the global label is anger.

## Slide 12 - Quantitative comparison (1 min 40)

The left part compares deletion AUC and insertion AUC.

LIME is the most convincing method in terms of faithfulness: it has the lowest deletion AUC and the highest insertion AUC. This means that the regions identified by LIME have a strong effect on model confidence.

The right part shows robustness. Grad-CAM and LIME are relatively stable, while SHAP is more fragile in our setup. This is not a general criticism of SHAP, but rather a limitation of our grid approximation and limited number of perturbations.

The conclusion of this slide is that there is a trade-off. LIME appears more faithful, Grad-CAM more stable and readable, and SHAP less reliable in this configuration.

## Slide 13 - Facial masking (1 min 20)

Masking lets us test human-interpretable regions: eyes, mouth and background.

The main result is that masking the mouth causes the largest average confidence drop and changes the prediction in 57 percent of analyzed cases. This is consistent with FER, because the mouth carries strong information for emotions such as happy, disgust, sad or anger.

However, the background also has a non-zero effect, which reminds us that the model may sometimes be influenced by non-facial regions. This is important when discussing model reliability.

## Slide 14 - Discussion (1 min 20)

The strengths of the project are the complete pipeline, the comparison of three XAI methods, and the combination of qualitative and quantitative evaluation.

The main results are: LIME is the most faithful on our sample, Grad-CAM is stable and readable, and SHAP is less robust in our approximation.

But we must be careful: the full XAI analysis is computed on 7 images, because LIME and SHAP are expensive. LIME and SHAP regions are coarse because they are based on a grid. The eye, mouth and background masks are also approximate.

Therefore, our conclusion is deliberately balanced: we do not claim that LIME is always better, but that it is the most convincing method within our protocol.

## Slide 15 - Conclusion (1 min)

To conclude, XAI methods are not interchangeable. On the same model and the same images, they produce different explanations and different scores.

First message: LIME is the most faithful in our deletion/insertion tests. Second message: Grad-CAM is more stable and easier to interpret. Third message: error analysis helps us understand some model confusions.

As future work, we would extend the analysis to more images, use more anatomical superpixels instead of a regular grid, and train a variant with explanation-guided regularization.

## Backup slides

Use them only if asked.

- Backup per-class details: useful if asked why macro-F1 is lower than weighted-F1.
- Backup reproducibility: useful if asked how to reproduce the figures.
- Backup references: useful if asked about the connection to the papers.

## Likely questions and short answers

**Why only 7 images for the XAI analysis?**
Because LIME and SHAP are computationally expensive: each image requires many perturbations and forward passes. We present this part as a focused analysis, not as an exhaustive statistical conclusion.

**Why not use a pretrained model?**
The project aims to analyze XAI in a controlled setting. A CNN trained from scratch avoids importing biases from a pretrained backbone and follows the initial project constraint.

**Is SHAP bad?**
No. In our setup, KernelSHAP is approximated on a grid with a limited number of perturbations. The results mainly show that this approximation is less stable here.

**Why does the mouth matter so much?**
This is consistent with FER: many emotions are distinguished by the mouth shape. But it can also cause errors when a local cue, such as an apparent smile, contradicts the global label.

**What would improve the project?**
Extending XAI evaluation to more images, using a more natural face segmentation, and testing a regularization term that encourages the model to focus on meaningful facial regions.
