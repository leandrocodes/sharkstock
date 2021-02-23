<template>
  <div class="tensorflow">
    <h1>Tensorflow</h1>
    <camera v-model="image" :style="style" @predict="predict"></camera>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import camera from '../components/Camera'
import { MODEL_CLASSES } from '../keras/classes'

export default {
  components: {
    camera
  },
  data: () => ({
    model: null,
    image: '',
    style: {
      border: '2px solid red',
      maxWidth: '100%'
    }
  }),
  async created() {
    // console.log(this.modelClasses)
    this.model = await tf.loadLayersModel(
      'http://127.0.0.1:9090/src/keras/model.json'
    )
  },
  methods: {
    async predict() {
      const img = new Image(100, 100)
      img.src = this.image

      const tensor = tf.browser
        .fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()

      const imageS = tensor
        .sub(tf.scalar(127.5))
        .div(tf.scalar(127.5))
        .expandDims()

      const res = await this.model.predict(imageS).data()
      const prediction = await this.getTopKClasses(res)

      console.log(prediction)
    },
    async getTopKClasses(values) {
      const valuesAndIndices = []
      for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({ value: values[i], index: i })
      }
      valuesAndIndices.sort((a, b) => {
        return b.value - a.value
      })
      const topkValues = new Float32Array(5)
      const topkIndices = new Int32Array(5)
      for (let i = 0; i < 5; i++) {
        topkValues[i] = valuesAndIndices[i].value
        topkIndices[i] = valuesAndIndices[i].index
      }

      const topClassesAndProbs = []
      for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
          className: MODEL_CLASSES[topkIndices[i]],
          probability: (topkValues[i] * 100).toFixed(2)
        })
      }
      return topClassesAndProbs
    }
  }
}
</script>
