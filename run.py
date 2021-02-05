#requires Tensorflow >= 1.15

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from sys import argv
import os, os.path

tf.enable_eager_execution() 

class ENCODER(tf.keras.Model):
  def __init__(self, embed_dim):
    super(ENCODER,self).__init__()
    self.den = tf.keras.layers.Dense(embed_dim)

  def call(self, x): 
    x = self.den(x)  
    x = tf.nn.relu(x)
    return x

class ATTEND(tf.keras.Model): 
  def __init__(self, units):
    super(ATTEND, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, feat, hidden): 
    hidden_ = tf.expand_dims(hidden, axis=1) 
    score = tf.nn.tanh( self.W1(feat) + self.W2(hidden_) )
    att_wt = tf.nn.softmax(self.V(score), axis=1) 
    context = att_wt*feat 
    context = tf.reduce_sum(context, axis=1) 

    return context, att_wt

class DECODER(tf.keras.Model):

  def __init__(self, units, embed_M, sentence_length):
    super(DECODER, self).__init__()
    self.units = units
    self.embed = tf.keras.layers.Embedding(input_dim=embed_M.shape[0], output_dim=embed_M.shape[1], weights=[embed_M], input_length=sentence_length, trainable=False )
    #self.embed = tf.keras.layers.Embedding(input_dim=embed_M.shape[0], output_dim=embed_M.shape[1], embeddings_initializer=tf.keras.initializers.Constant(embed_M), input_length=sentence_length, trainable=True)
    self.lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, return_state=True) 
    self.den1 = tf.keras.layers.Dense(units)
    self.den2 = tf.keras.layers.Dense(embed_M.shape[0])   
    self.attend = ATTEND(units)

  def call(self, tok, feat, hidden): 
    context, att_wt = self.attend(feat, hidden)
    x = self.embed(tok)
    context_ = tf.expand_dims(context,1) 
    x = tf.concat([context_, x], axis=2) 
    output, state, _ = self.lstm(x) 
    x = self.den1(output)
    x = tf.reshape(x, (-1, x.shape[2])) 
    x = self.den2(x) 
    return x, state, att_wt

  def reset_state(self, batch_size): 
    return tf.zeros((batch_size, self.units))



def feat_extract():

  IV3 = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet') 

  x_in = IV3.input 
  x_out= IV3.layers[-1].output

  return tf.keras.Model(inputs=x_in, outputs=x_out)

def load_image(arg):
    img = tf.io.read_file(arg)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, arg


max_length = 80
def evaluate(image): 
    attention_plot = np.zeros((max_length, 64))

    hidden = dec.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = IV3_feat(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = enc(img_tensor_val)

    dec_input = tf.expand_dims([word_ind_map['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = dec(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(ind_word_map[predicted_id])

        if ind_word_map[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0) 

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot



def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10),dpi=200)

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


curr_dir = os.getcwd()





M = np.load(os.path.join(curr_dir,'ImageCap/embedB.npy'))
cap_seq = np.load('ImageCap/caption_vec.npy')

word_ind_map=dict()
with open(os.path.join(curr_dir,'ImageCap/word_ind_map.pkl'), 'rb') as f:
  word_ind_map = pickle.load(f)

ind_word_map=dict()
with open(os.path.join(curr_dir,'ImageCap/ind_word_map.pkl'), 'rb') as f:
  ind_word_map = pickle.load(f)

IV3_feat = tf.keras.models.load_model(os.path.join(curr_dir,'ImageCap/IV3_feat.h5'))

image_path = argv[1]
version = argv[2]


if int(version)==1 or int(version)>5:
    print('Wrong model version, select a version from 2 to 5')
else:
    embed_dim = 300
    units = 512
    enc = ENCODER(embed_dim)
    dec = DECODER(units, M, 80)
    enc.load_weights(os.path.join(curr_dir,'ImageCap/models/encoder'+str(version)+'/'))
    dec.load_weights(os.path.join(curr_dir,'ImageCap/models/decoder'+str(version)+'/'))


    result, attention_plot = evaluate(image_path)
    bl = True 
    while(bl):
      if(len(result)<=20):
        bl =False
      else:
        result, attention_plot = evaluate(image_path)
        
    print ('Prediction Caption:', ' '.join(result))
    #plot_attention(image_path, result, attention_plot)