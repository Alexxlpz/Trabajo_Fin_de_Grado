import React from 'react';
import { Text, View, StyleSheet } from "react-native";
import Camera from '../component/Camera';

const Recording = () => {
    return (
        <Camera/>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center'
    },
    title: {
        fontSize: 20
    }
});

export default Recording;
