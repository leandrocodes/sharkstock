<template>
  <section :style="styling" v-if="isValid" class="photo-capture">
    <video
      v-show="showVideo"
      ref="player"
      class="camera"
      autoplay
      playsinline
    />
    <canvas v-show="!showVideo" class="preview" ref="canvas" />
    <div v-if="!hideBtns" class="center photo-capture-actions">
      <button
        :class="'btn flex-center ' + buttonsClasses"
        @click.prevent="capture"
        v-if="showVideo"
      >
        {{ captureBtnContent }}
      </button>
      <div class="controls" v-else>
        <button :class="'btn ' + buttonsClasses" @click.prevent="cancel">
          {{ cancelBtnContent }}
        </button>
        <button :class="'btn ' + buttonsClasses" @click.prevent="done">
          {{ doneBtnContent }}
        </button>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'Camera',
  props: {
    hideBtns: {
      type: Boolean,
      isRequired: false,
      default: false
    },
    styling: {
      type: Object,
      isRequired: false
    },
    value: {
      default: null
    },
    hideButtons: {
      type: Boolean,
      default: false
    },
    buttonsClasses: {
      type: String,
      default: ''
    },
    captureBtnContent: {
      default: 'Capture'
    },
    cancelBtnContent: {
      default: 'Cancel'
    },
    doneBtnContent: {
      default: 'OK'
    }
  },
  data() {
    return {
      showVideo: true,
      picture: null,
      isValid: true
    }
  },
  mounted() {
    this.videoPlayer = this.$refs.player
    this.canvasElement = this.$refs.canvas
    this.streamUserMediaVideo()
  },
  methods: {
    async streamUserMediaVideo() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        return
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: 'user',
            width: window.innerWidth
          }
        })
        this.videoPlayer.srcObject = stream
      } catch (_error) {
        this.invalid = false
      }
    },
    capture() {
      this.showVideo = false
      this.canvasElement.width = this.videoPlayer.videoWidth
      this.canvasElement.height = this.videoPlayer.videoHeight
      const context = this.canvasElement.getContext('2d')
      context.translate(this.canvasElement.width, 0)
      context.scale(-1, 1)
      context.drawImage(this.$refs.player, 0, 0)
      this.stopVideoStream()
      this.picture = this.$refs.canvas.toDataURL()
    },
    done() {
      this.$emit('input', this.picture)
      this.$emit('predict')
      this.showVideo = true
      this.streamUserMediaVideo()
    },
    cancel() {
      this.showVideo = true
      this.streamUserMediaVideo()
    },
    stopVideoStream() {
      if (!(this.$refs.player && this.$refs.player.srcObject)) return
      this.$refs.player.srcObject.getVideoTracks().forEach(track => {
        track.stop()
      })
    }
  }
}
</script>
