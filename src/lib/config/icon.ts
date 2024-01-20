import type { Icon } from '$lib/types/icon'
import { site } from '$lib/config/site'

// }

export const favicon: Icon = {
  src: '/favicon.png',
  sizes: '48x48',
  type: 'image/png'
}

export const any: { [key: number]: Icon } = {
  180: {
    src: '/assets/any@180.png',
    sizes: '180x180',
    type: 'image/png'
  },
  192: {
    src: '/assets/any@192.png',
    sizes: '192x192',
    type: 'image/png'
  },
  512: {
    src: '/assets/any@512.png',
    sizes: '512x512',
    type: 'image/png'
  }
}

export const maskable: { [key: number]: Icon } = {
  192: {
    src: '/assets/any@192.png',
    sizes: '192x192',
    type: 'image/png'
  },
  512: {
    src: '/assets/any@512.png',
    sizes: '512x512',
    type: 'image/png'
  }
}

