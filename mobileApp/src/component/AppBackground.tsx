import React, { ReactNode } from 'react';
import { ImageBackground, StyleSheet, View, StyleProp, ViewStyle } from 'react-native';

type AppBackgroundProps = {
    children: ReactNode;
    contentStyle?: StyleProp<ViewStyle>;
    backgroundColor?: string;
    blurRadius?: number;
};

export default function AppBackground({
    children,
    contentStyle,
    backgroundColor = '#8BB39B',
    blurRadius = 4,
}: AppBackgroundProps) {
    return (
        <View style={[styles.container, { backgroundColor }]}>
            <ImageBackground
                source={require('../../assets/home_background.jpg')}
                style={styles.background}
                imageStyle={styles.image}
                blurRadius={blurRadius}
            >
                <View style={[styles.content, contentStyle]}>{children}</View>
            </ImageBackground>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        width: '100%',
        height: '100%',
    },
    background: {
        flex: 1,
        width: '100%',
        height: '100%',
    },
    image: {
        backgroundColor: 'transparent',
    },
    content: {
        flex: 1,
    },
});