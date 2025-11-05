import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Fontisto } from '@expo/vector-icons';
import Constants from "expo-constants";

export default function Camera() {
    const [recording, setRecording] = useState<boolean>(false);
    const [permission, requestPermission] = useCameraPermissions();

    if (!permission) {
        // Camera permissions are still loading.
        return <View />;
    }

    if (!permission.granted) {
        // Camera permissions are not granted yet.
        return (
            <View style={styles.container}>
                <Text style={styles.message}>We need your permission to show the camera</Text>
                <Button onPress={requestPermission} title="grant permission" />
            </View>
        );
    }

    function toggleCameraRecording(){
        setRecording(current => (current === false));
    }

    return (
        <View style={styles.container}>
            <CameraView style={styles.camera} active={recording} />
            <View style={styles.recordingText}>
                <View style={{ backgroundColor: recording ? 'red' : 'gray', borderRadius: 4, flexDirection: 'row', paddingHorizontal: 4 }}>
                    <Fontisto name="record" size={10} color='white' style={{marginRight: 6, marginTop: 5}} />
                    <Text>{recording ? 'Camera...' : 'Not Camera'}</Text>
                </View>
            </View>
            <View style={styles.buttonContainer}>
                <TouchableOpacity style={recording ? styles.button_recording : styles.button_not_recording}
                    onPress={toggleCameraRecording}>

                    <Fontisto name="record" size={24} color={recording ? 'white' : 'black'} />

                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
    },
    message: {
        textAlign: 'center',
        paddingBottom: 10,
    },
    camera: {
        flex: 1,
    },
    buttonContainer: {
        position: 'absolute',
        bottom: 64,
        flexDirection: 'row',
        backgroundColor: 'transparent',
        width: '100%',
        paddingHorizontal: 64,
        justifyContent: 'center',
        alignItems: 'center',
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
    button_recording: {
        backgroundColor: 'red',
        borderRadius: 50,
        padding: 30,
        alignItems:'center',
        alignSelf: 'center'
    },
    button_not_recording: {
        backgroundColor: 'white',
        borderRadius: 50,
        padding: 30,
        alignItems:'center',
        alignSelf: 'center'
    },
    recordingText: {
        position: 'absolute',
        top: Constants.statusBarHeight + 5,
        right: 0,
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 8,
        zIndex: 10,
        flexDirection: 'row',
    },
});