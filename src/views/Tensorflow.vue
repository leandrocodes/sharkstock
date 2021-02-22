<template>
  <div class="tensorflow">
    <h1>Tensorflow</h1>
    <photo-capture v-model="image"></photo-capture>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import { PhotoCapture } from 'vue-media-recorder'
export default {
  components: {
    PhotoCapture
  },
  data: () => ({
    model: null,
    image: ''
  }),
  async created() {
    this.model = await tf.loadLayersModel(
      'http://127.0.0.1:9090/src/keras/model.json'
    )
  }
}
</script>
