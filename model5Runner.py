"NN Training Playground"
import cv2
import numpy as np
import tensorflow as tf
import tensorflowhelper as tfh
import inputpreprocessor.Data3 as Data
import inputpreprocessor.ObjClass2 as ObjClass

def half_sigmoid(input):
    input_bypass, input_to_process = tf.split(3, 2, input)

    input_processed = tf.sigmoid(input_to_process)

    output = tf.concat(3, [input_bypass, input_processed])

    return output

def main():
    """Entry point function"""
    model_name = "model5-b"
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

    data = Data.DataFeeder("data/", dynamic_load=True,
                           filename=model_name+"-FeedCache", data_padding=0,
                           label_height=512, label_width=640)

    batch = data.get_batch(5)

    life.connect_neural_network(sample_input=batch[0], will_train=False)

    # life.load_saved_model(model_name+"-save-data")
    life.load_saved_model("auto-save/"+model_name+"-save-data18.0")
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
            cv2.waitKey(50)

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
