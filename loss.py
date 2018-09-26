# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    #print(len(values))
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    use_embedding   = True,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    print(G.input_shapes[0][1:])
    print(latents)
    labels, embeddings = training_set.get_random_labels_tf(minibatch_size)
    #labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, embeddings, is_training=True)
    if(use_embedding):
        fake_scores_out, fake_labels_out, fake_embeddings_out = fp32(D.get_output_for(fake_images_out, labels, embeddings, is_training=True))
    else:
        fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, labels, embeddings, is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            #label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_fakes = tf.losses.mean_squared_error(labels, fake_labels_out)
        loss += label_penalty_fakes * cond_weight

        if(use_embedding):
            with tf.name_scope('EmbeddingPenalty'):
                embedding_penalty_fakes = tf.losses.mean_squared_error(embeddings, fake_embeddings_out)
            loss += embedding_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, embeddings,
    use_embedding   = True,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, embeddings, is_training=True)
    if(use_embedding):
        real_scores_out, real_labels_out, real_embeddings_out = fp32(D.get_output_for(reals, labels, embeddings, is_training=True))
        fake_scores_out, fake_labels_out, fake_embeddings_out = fp32(D.get_output_for(fake_images_out, labels, embeddings, is_training=True))
    else:
        real_scores_out, real_labels_out = fp32(D.get_output_for(reals, labels, embeddings, is_training=True))
        fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, labels, embeddings, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        if(use_embedding):
            mixed_scores_out, mixed_labels_out, mixed_embeddings_out = fp32(D.get_output_for(mixed_images_out, labels, embeddings, is_training=True))
        else:
            mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, labels, embeddings, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            #label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            #label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tf.losses.mean_squared_error(labels, real_labels_out)
            label_penalty_fakes = tf.losses.mean_squared_error(labels, fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
        if(use_embedding):
            with tf.name_scope('EmbeddingPenalty'):
                embedding_penalty_reals = tf.losses.mean_squared_error(embeddings, real_embeddings_out)
                embedding_penalty_fakes = tf.losses.mean_squared_error(embeddings, fake_embeddings_out)
                embedding_penalty_reals = tfutil.autosummary('Loss/embedding_penalty_reals', embedding_penalty_reals)
                embedding_penalty_fakes = tfutil.autosummary('Loss/embedding_penalty_fakes', embedding_penalty_fakes)
            loss += (embedding_penalty_reals + embedding_penalty_fakes) * cond_weight

    return loss

#----------------------------------------------------------------------------
