"NN Training Playground"
import cv2
import numpy as np
import tensorflow as tf
import tensorflowhelper as tfh
import inputpreprocessor.Data3 as Data
import inputpreprocessor.ObjClass2 as ObjClass

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

    model_name = "model4-a"
    padding_size = 5
    train_data_height = 80
    train_data_width = 80
    batch_size = 60
    total_number_of_iteration = 3000

    life = tfh.Life(
        tfh.NeuralNetwork(
            layers=[
                tfh.ValidationLayer(shape=[None, train_data_height + padding_size*2, train_data_width + padding_size*2, 3], dtype=tf.uint8),
                tfh.OpLayer(tf.to_float),
                tfh.OpLayer(lambda x: x/255. -0.5),
                tfh.ConvLayer(kernel_width=3, depth_out=10, depth_in=3, padding=False),

                tfh.ConvLayer(kernel_width=3, depth_out=10, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=20, padding=False),
                tfh.ConvLayer(kernel_width=3, depth_out=10, padding=False),

                tfh.ConvLayer(kernel_width=3, depth_out=2, padding=False),
                tfh.ValidationLayer(shape=[None, train_data_height, train_data_width, 2], dtype=tf.float32),
                # tfh.ReshapeLayer(shape=[None, 2]),
            ]
        )
    )

    data = Data.DataFeeder("data/", dynamic_load=True,
                           filename=model_name+"-TrainCache", data_padding=padding_size,
                           label_height=train_data_height, label_width=train_data_width)

    batch = data.get_batch(batch_size, shuffle=True, slient=False)

    print(batch[0].dtype, batch[0].shape)
    print(batch[1].dtype, batch[1].shape)

    print(np.amax(batch[0], axis=(0, 1, 2)), np.amax(batch[1], axis=(0, 1, 2)))

    life.connect_neural_network(sample_input=batch[0]) 

    tf_flatten_label = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    tf_flatten_result = tf.reshape(life.tfvResult_pipe, shape=[-1, 2])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf_flatten_result, tf_flatten_label))

    trainer = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # life.load_saved_model("auto-save/"+model_name+"-save-data3.0")
    life.session.run(tf.initialize_all_variables())
    # life.load_saved_model(model_name+"-save-data")
    # life.init_var()

    for counter in range(total_number_of_iteration):
        batch = data.get_batch(batch_size, shuffle=True)
        preview = counter % 20 == 0
        process_list = [cross_entropy, trainer, life.tfvResult_pipe] if preview else [cross_entropy, trainer]
        result = life.feed(batch[0], process_list=process_list,
                           feed_dict={
                               tf_flatten_label:batch[1].reshape((-1, 2))})
        print(counter, "loss", result[0])
        if preview:
            # print(result[2][0].sum())
            # print(result[2][0])
            # for index in range(batch_size):
            for index in range(2):
                cv2.imshow("image-in", batch[0][index])
                cv2.imshow("hypo", ObjClass.combine_label(result[2][index]))
                cv2.imshow("expecr", ObjClass.combine_label(batch[1][index]))
                cv2.waitKey(30)
        if counter % 100 == 0:
            life.save_current_model("auto-save/"+model_name+"-sava-data"+str(counter/100))

    #for counter in range(100):

    #    batch = data.get_batch(60, shuffle=True)
    #    result = life.train(input_layer_value=batch[0], output_layer_value=batch[1])

    #    if counter%27 == 0:
    #        print("{} iteration".format(counter), result)
    #        data_in = batch[0]
    #        feed_result = life.feed(input_layer_value=data_in)
    #        hypo = feed_result[0]
    #        # print("HYPO SHAPE", hypo.shape)
    #        expect = batch[1][0]

    #    if counter%100 == 0:
    #        life.save_current_model("auto-save/"+model_name+"-save-data"+str(counter/100))

    life.save_current_model(model_name+"-save-data")

if __name__ == "__main__":
    try:
        main()
    except tfh.utilities.TFHError as tfh_error:
        print(tfh_error)
