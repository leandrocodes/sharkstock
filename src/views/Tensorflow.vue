<template>
  <h1>Tensorflow</h1>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
export default {
  data: () => ({
    model: null
  }),
  async created() {
    this.model = await tf.loadLayersModel(
      'http://127.0.0.1:9090/src/keras/model.json'
    )
  }
}
</script>
