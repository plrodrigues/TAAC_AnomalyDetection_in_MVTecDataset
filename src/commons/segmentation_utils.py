import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_scores(inputs, recons, idx, threshold=0.5):
    scores = np.sqrt(((inputs[idx] - recons[idx])**2).sum(-1)) # anomaly mask
    scores = (scores - scores.min())/(scores.max() - scores.min()) # Normalize scores
    scores_bin = np.where(scores > threshold, 1, 0) # Threshold scores
    return scores, scores_bin


def get_filled_score_map(scores, threshold):
    th, b_w_score = cv2.threshold(scores*255, int(255*threshold), 255, cv2.THRESH_BINARY_INV)
    b_w_score = b_w_score.astype(np.uint8)
    
    # Fill contours
    img_to_fill = cv2.bitwise_not(b_w_score)
    kernel = np.ones((5, 5), np.uint8)
    img_to_fill_dilated = cv2.dilate(img_to_fill,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(img_to_fill_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(img_to_fill_dilated, [cnt], 0, 255, -1)
    
    img_to_fill = cv2.erode(img_to_fill_dilated,kernel,iterations = 1)
    img_filled = cv2.bitwise_not(img_to_fill)
    return img_filled
    
    
def plot_input_gt_score_bw_mask(inputs_resh, targets_resh, scores, b_w_score, img_filled, idx):
    # Compare
    fig, axs = plt.subplots(1, 5, figsize=(8, 8))
    axs[0].imshow(inputs_resh[idx])
    axs[0].set_title('Raw image')
    axs[0].axis('off')
    axs[1].imshow(targets_resh[idx])
    axs[1].set_title('Ground Truth image')
    axs[1].axis('off')
    axs[2].imshow(scores.numpy()*255, cmap="Greys_r")
    axs[2].set_title('Gray score')
    axs[2].axis('off')
    axs[3].imshow(b_w_score, cmap="Greys_r")
    axs[3].set_title('BlackWhite score')
    axs[3].axis('off')
    axs[4].imshow(img_filled, cmap="Greys_r")
    axs[4].set_title('Filled score')
    axs[4].axis('off')
    fig.tight_layout()
    plt.show()