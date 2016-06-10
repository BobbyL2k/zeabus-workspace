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
                tfh.ValidationLayer(shape=[None, 512/8 + 4, 640/8 + 4, 3], dtype=tf.uint8),
                tfh.OpLayer(tf.to_float),
                tfh.OpLayer(lambda x: x/255.),
                tfh.ConvLayer(kernel_width=3, depth_out=30, depth_in=3, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=9, padding=False),
                tfh.ValidationLayer(shape=[None, 512/8, 640/8, 9], dtype=tf.float32),
            ]
        )
    )

    data = Data.DataFeeder("data/",
                           filename="TrainCache", data_padding=2,
                           label_height=64, label_width=80)

    batch = data.get_batch(100)

    print(batch[0].dtype, batch[0].shape)
    print(batch[1].dtype, batch[1].shape)

    life.connect_neural_network(sample_input=batch[0], sample_output=batch[1], will_train=True)

    life.load_saved_model("model1")
    # life.init_var()

    for counter in range(20):

        batch = data.get_batch(100)
        result = life.train(input_layer_value=batch[0], output_layer_value=batch[1])

        if counter%27 == 0:
            print("{} iteration".format(counter), result)
            data_in = batch[0]
            feed_result = life.feed(input_layer_value=data_in)
            hypo = feed_result[0]
            # print("HYPO SHAPE", hypo.shape)
            expect = batch[1][0]
            cv2.imshow("image-in", data_in[0])
            cv2.imshow("hypo",  ObjClass.combine_label(hypo))
            cv2.imshow("expect", ObjClass.combine_label(expect))
            cv2.waitKey(20)

    life.save_current_model("model1")

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
