import React from 'react';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Text, View, StyleSheet, TouchableOpacity, Alert } from "react-native";
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useSession } from '../SessionContext';
import AppBackground from '../component/AppBackground';

type HomeScreenNavigationProp = NativeStackNavigationProp<any>;

const HomeScreen = ({ navigation }: { navigation: HomeScreenNavigationProp }) => {
    const { isLoggedIn } = useSession();

    return (
        <AppBackground>
            <View style={styles.overlay}>
                <View style={styles.container}>
                    <View style={styles.buttonsContainer}>

                        <TouchableOpacity
                            style={styles.primaryButton}
                            onPress={isLoggedIn ? () => navigation.navigate('Recording') : () => Alert.alert('Acceso Denegado', 'Debes iniciar sesión para acceder a la cámara')}
                        >
                            <Ionicons name="camera-outline" size={22} color="#FFFFFF" style={styles.buttonIcon} />
                            <Text style={styles.primaryButtonText}>Cámara</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.secondaryButton}
                            onPress={isLoggedIn ? () => navigation.navigate('Uploading') : () => Alert.alert('Acceso Denegado', 'Debes iniciar sesión para acceder a la subida de archivos')}
                        >
                            <Ionicons name="cloud-upload-outline" size={22} color="#00875A" style={styles.buttonIcon} />
                            <Text style={styles.secondaryButtonText}>Subir Archivo</Text>
                        </TouchableOpacity>

                    </View>
                </View>
            </View>
        </AppBackground>
    );
}

const styles = StyleSheet.create({
    background: {
        flex: 1,
        width: '100%',
        height: '100%',
    },
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(255, 255, 255, 0.65)', // Superposición blanca translúcida
    },
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 30,
    },
    topSection: {
        alignItems: 'center',
        marginBottom: 80, // Separación entre títulos y botones
    },
    iconCircle: {
        width: 110,
        height: 110,
        borderRadius: 55,
        backgroundColor: '#D1F2E0', // Verde muy claro
        borderWidth: 3,
        borderColor: '#E8F8F0', // Borde casi blanco
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 15,
        // Sombra
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 5,
        elevation: 5,
    },
    title: {
        fontSize: 32,
        fontWeight: '900',
        color: '#004D40',
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 15,
        fontWeight: '600',
        color: '#007A5E',
    },
    buttonsContainer: {
        width: '100%',
        gap: 18, // Espaciado nativo entre botones
    },
    primaryButton: {
        flexDirection: 'row',
        backgroundColor: '#00875A',
        paddingVertical: 16,
        borderRadius: 30,
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 3,
        elevation: 3,
    },
    primaryButtonText: {
        color: '#FFFFFF',
        fontSize: 16,
        fontWeight: 'bold',
    },
    secondaryButton: {
        flexDirection: 'row',
        backgroundColor: '#FFFFFF',
        paddingVertical: 16,
        borderRadius: 30,
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: '#00875A',
    },
    secondaryButtonText: {
        color: '#00875A',
        fontSize: 16,
        fontWeight: 'bold',
    },
    buttonIcon: {
        marginRight: 8,
    }
});

export default HomeScreen;