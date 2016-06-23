"NN Training Playground"
import cv2
import numpy as np
import tensorflow as tf
import tensorflowhelper as tfh
import inputpreprocessor.Data2 as Data
import inputpreprocessor.ObjClass as ObjClass

def conv_cross_entropy(hypo, actual_value):
    """Calculate Cross Entropy
    Args:
        hypo         -- TensorFlow variable of the hypothesis
        actual_value -- TensorFlow variable of the expected value
    Returns:
        TensorFlow variable of the Cross Entropy
    """
    return -tf.reduce_mean(
        actual_value * tf.log(tf.clip_by_value(hypo, 1e-10, 1.0)) +
        (1-actual_value) * tf.log(tf.clip_by_value(1-hypo, 1e-10, 1.0)))

def main():
    """Entry point function"""
    model_name = "model2-d"
    life = tfh.Life(
        tfh.NeuralNetwork(
            layers=[
                tfh.ValidationLayer(shape=[None, 96 + 8, 120 + 8, 3], dtype=tf.uint8),
                tfh.OpLayer(tf.to_float),
                tfh.OpLayer(lambda x: x/255.),
                tfh.ConvLayer(kernel_width=3, depth_out=30, depth_in=3, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=50, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=40, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=9, padding=False),
                tfh.OpLayer(tf.sigmoid),
                tfh.ValidationLayer(shape=[None, 96, 120, 9], dtype=tf.float32),
            ]
        ), cost_function=conv_cross_entropy, optimizer=tf.train.AdamOptimizer(0.001)
    )

    data = Data.DataFeeder("data/", dynamic_load=False,
                           filename="TrainCache", data_padding=4,
                           label_height=96, label_width=120)

    batch = data.get_batch(60)

    print(batch[0].dtype, batch[0].shape)
    print(batch[1].dtype, batch[1].shape)

    life.connect_neural_network(sample_input=batch[0], sample_output=batch[1], will_train=True)

    # life.load_saved_model("auto-save/"+model_name+"-save-data3.0")
    # life.load_saved_model(model_name+"-save-data")
    life.init_var()

    for counter in range(100):

        batch = data.get_batch(60, shuffle=True)
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

        if counter%100 == 0:
            life.save_current_model("auto-save/"+model_name+"-save-data"+str(counter/100))

    life.save_current_model(model_name+"-save-data")

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
