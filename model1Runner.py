"NN Training Playground"
import cv2
import numpy as np
import tensorflow as tf
import tensorflowhelper as tfh
import inputpreprocessor.Data2 as Data
import inputpreprocessor.ObjClass as ObjClass

def main():
    """Entry point function"""
    life = tfh.Life(
        tfh.NeuralNetwork(
            layers=[
                tfh.ValidationLayer(shape=[None, 512, 640, 3], dtype=tf.uint8),
                tfh.OpLayer(tf.to_float),
                tfh.OpLayer(lambda x: x/255.),
                tfh.ConvLayer(kernel_width=3, depth_out=30, depth_in=3),
                tfh.ConvLayer(kernel_width=3, depth_out=9, ),
                tfh.ValidationLayer(shape=[None, 512, 640, 9], dtype=tf.float32),
            ]
        )
    )

    data = Data.DataFeeder("data/",
                           filename="FeedCache", data_padding=0,
                           label_height=512, label_width=640)

    batch = data.get_batch(5)

    life.connect_neural_network(sample_input=batch[0], will_train=False)

    life.load_saved_model("model1")
    # life.init_var()

    print(batch[0].dtype, batch[0].shape)
    print(batch[1].dtype, batch[1].shape)

    ##
    data_in = batch[0]
    feed_result = life.feed(input_layer_value=data_in)
    hypo = feed_result[0]
    print("HYPO SHAPE", hypo.shape)

    for _ in range(2):
        for index, (img, expect_label) in enumerate(zip(batch[0], batch[1])):
            cv2.imshow("image-in", img)
            cv2.imshow("hypo", ObjClass.combine_label(feed_result[index]))
            cv2.imshow("expect", ObjClass.combine_label(expect_label))
            cv2.waitKey(0)

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
