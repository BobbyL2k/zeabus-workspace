"NN Training Playground"
import cv2
import numpy as np
import tensorflow as tf
import tensorflowhelper as tfh
import inputpreprocessor.GeneralDataFeeder as Data
import inputpreprocessor.ObjClass2 as ObjClass

def half_sigmoid(input):
    input_bypass, input_to_process = tf.split(3, 2, input)

    input_processed = tf.sigmoid(input_to_process)

    output = tf.concat(3, [input_bypass, input_processed])

    return output

def raw_preprocess(img):
    img = cv2.resize(img, (640, 512))
    cv2.GaussianBlur(img, None, 3, img)

    b, g, r = cv2.split(img)
    cv2.equalizeHist(r, r)
    cv2.equalizeHist(g, g)
    cv2.equalizeHist(b, b)
    cv2.merge([b, g, r], img)

    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)

    h, s, v = cv2.split(img)
    cv2.equalizeHist(s, s)
    cv2.equalizeHist(v, v)
    cv2.merge([h, s, v], img)
    cv2.cvtColor(img, cv2.COLOR_HSV2BGR, img)

    return img

def label_preprocess(label_img):
    label_img = cv2.resize(label_img, (640, 512))
    return label_img

def main():
    """Entry point function"""
    model_name = "model6-a"
    padding_size = 0
    train_data_height = 512
    train_data_width = 640
    batch_size = 5

    life = tfh.Life(
        tfh.NeuralNetwork(
            layers=[
                tfh.ValidationLayer(shape=[None, train_data_height + padding_size*2, train_data_width + padding_size*2, 3], dtype=tf.uint8),
                tfh.OpLayer(tf.to_float),
                tfh.OpLayer(lambda x: x/255. -0.5),
                tfh.ConvLayer(kernel_width=3, depth_out=10, depth_in=3, padding=True),
                tfh.OpLayer(half_sigmoid),

                tfh.ConvLayer(kernel_width=3, depth_out=10, padding=True),
                tfh.OpLayer(half_sigmoid),
                tfh.ConvLayer(kernel_width=3, depth_out=20, padding=True),
                tfh.OpLayer(half_sigmoid),
                tfh.ConvLayer(kernel_width=3, depth_out=10, padding=True),
                tfh.OpLayer(half_sigmoid),

                tfh.ConvLayer(kernel_width=3, depth_out=2, padding=True),
                tfh.ValidationLayer(shape=[None, train_data_height, train_data_width, 2], dtype=tf.float32),
                # tfh.ReshapeLayer(shape=[None, 2]),
            ]
        )
    )

    data = Data.DataFeeder("data/", dynamic_load=True, raw_preprocess=raw_preprocess, label_preprocess=label_preprocess,
                           cache_name=model_name+"-TrainCache", data_padding=padding_size,
                           label_height=train_data_height, label_width=train_data_width)

    batch = data.get_batch(5)

    life.connect_neural_network(sample_input=batch[0], will_train=False)

    # life.load_saved_model(model_name+"-save-data")
    life.load_saved_model("auto-save/"+model_name+"-save-data15.0")
    # life.init_var()

    print(batch[0].dtype, batch[0].shape)
    print(batch[1].dtype, batch[1].shape)

    ##

    # data.get_batch(500)

    for _ in range(200):
        batch = data.get_batch(5, shuffle=False)
        data_in = batch[0]
        feed_result = life.feed(input_layer_value=data_in)
        hypo = feed_result[0]
        print("HYPO SHAPE", hypo.shape)
        for index, (img, expect_label) in enumerate(zip(batch[0], batch[1])):
            cv2.imshow("image-in", img)
            cv2.imshow("hypo", ObjClass.combine_label(feed_result[index]))
            # print(feed_result[index])
            cv2.imshow("expect", ObjClass.combine_label(expect_label))
            cv2.waitKey(30)

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
