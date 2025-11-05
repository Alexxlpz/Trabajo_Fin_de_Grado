import theme from '../theme.js';
import React from 'react'
import { StyleSheet, Text} from 'react-native';


const styles = StyleSheet.create({
    text: {
        fontSize: 16,
        fontFamily: 'System',
    },
    big: {
        fontSize: theme.fontSizes.big,
    },
    medium: {
        fontSize: theme.fontSizes.medium,
    },
    small: {
        fontSize: theme.fontSizes.small,
    },
    bold: {
        fontWeight: 'bold',
    }
})



export default function StyledText({ children, size, bold, dark, light, style, ...restProps }) {
    const textStyles = [
        styles.text,
        size === 'big' && styles.big,
        size === 'medium' && styles.medium,
        size === 'small' && styles.small,
        bold && { fontWeight: 'bold' },
        restProps.style,
    ]

    return (
        <Text style={textStyles} {...restProps}>
            {children}
        </Text>
    );
}