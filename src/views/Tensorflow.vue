<template>
  <div class="tensorflow">
    <h1>Tensorflow</h1>
    <camera v-model="image"></camera>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import camera from '../components/Camera'
export default {
  components: {
    camera
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
