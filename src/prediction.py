import pandas as pd
import numpy as np
import cv2
import torch
import glob as glob
import PIL.Image
from model import create_model
from config import NUM_CLASSES, CLASSES, TEST_DIR, OUT_DIR, DETECTION_THRESHOLD

def predict_objects(model_name, image_name, detection_threshold=DETECTION_THRESHOLD, visualise=False, save=False):
    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load the model and the trained weights
    model = create_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('../outputs/{}'.format(model_name), map_location=device))
    model.eval()

    # directory where all the images are present
    test_images = glob.glob(f"{TEST_DIR}/{image_name}*")
    print(f"Test instances: {len(test_images)}")

    # read in image
    image = cv2.imread(test_images[0])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        results = model(image)

    # visualise picture
    if visualise:
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in results]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j],
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2, lineType=cv2.LINE_AA)
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            if save:
                cv2.imwrite(f"../test_predictions/{image_name}.jpg", orig_image, )
            cv2.destroyAllWindows()

    # return results as df for all instances above detection_threshold
    results = results[0]
    results_df = pd.DataFrame(results['boxes'], columns=['x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright'])
    results_df['labels'] = results['labels']
    results_df['scores'] = results['scores']
    results_df['width'] = results_df['x_bottomright'] - results_df['x_topleft']
    results_df['height'] = results_df['y_bottomright'] - results_df['y_topleft']

    if detection_threshold == None:
        return results_df, orig_image.shape

    else:
        return results_df.loc[results_df['scores'] > detection_threshold], orig_image.shape

def create_cutting_file(file_name, results, image_shape):
    # create jpg with cv2
    img = np.zeros(list(image_shape), dtype=np.uint8)
    img.fill(255)

    # fill with rectangles for object detection/cutting
    cv2.rectangle(img,
                      (int(min(results['x_topleft'])), int(max(results['y_topleft']))),
                      (int(max(results['x_bottomright'])), int(min(results['y_bottomright']))),
                      (0, 0, 255), 2)
    #for index, row in results.iterrows():
    #    cv2.rectangle(img,
    #                  (int(row['x_topleft']), int(row['y_topleft'])),
    #                  (int(row['x_bottomright']), int(row['y_bottomright'])),
    #                  (0, 0, 255), 2)
    cv2.imwrite(f"{OUT_DIR}/cutting_file/{file_name}.jpg", img)

    # convert to pdf
    img_pdf = PIL.Image.open(f"{OUT_DIR}/cutting_file/{file_name}.jpg")
    img_pdf = img_pdf.convert('RGB')
    img_pdf.save(f"{OUT_DIR}/cutting_file/{file_name}.pdf")

if __name__ == '__main__':
    # define parameters
    model_name = 'output_03_epoch10/model10.pth'
    detection_threshold = 0.8  # any detection having score below this will be discarded
    image_name = 'image_test'

    # call functions
    results, image_shape = predict_objects(model_name, image_name, detection_threshold, return_type='dict',
                                           visualise=True)
    create_cutting_file('test', results, image_shape)
