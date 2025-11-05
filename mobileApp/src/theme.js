const theme = {
    light: {
        background: '#ffffff',
        text: '#000000',
        primary: '#1E90FF',
        secondary: '#FF69B4'
    },
    dark: {
        background: '#121212',
        text: '#ffffff',
        primary: '#BB86FC',
        secondary: '#03DAC6'
    },
    fontSizes: {
        big: 32,
        medium: 24,
        small: 10
    },
}

export type theme = typeof theme;
export default theme;