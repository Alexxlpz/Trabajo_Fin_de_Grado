import React, { useState } from 'react';
import {
	View,
	Text,
	TextInput,
	TouchableOpacity,
	StyleSheet,
	ScrollView,
	KeyboardAvoidingView,
	Platform,
    Alert,
} from 'react-native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useNavigation } from '@react-navigation/native';
import { IP_ADDRESS } from '@env';
import { useSession } from '../SessionContext';
import AppBackground from '../component/AppBackground';

type RootStackParamList = {
	Home: undefined;
	Login: undefined;
	Register: undefined;
};

type RegisterScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>;

const RegisterScreen = () => {
	const navigation = useNavigation<RegisterScreenNavigationProp>();
	const { setRecents, setIsLoggedIn, setUser } = useSession();
	const [name, setName] = useState('');
	const [email, setEmail] = useState('');
	const [password, setPassword] = useState('');

    const handleRegister = async (name:String, email: string, password: string) => {
        try {
            const response = await fetch(`http://${IP_ADDRESS}:8000/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                name: name,
                email: email,
                password: password
                }),
            });
            
            if (!response.ok) {
                Alert.alert('Error', `La respuesta del servidor no es correcta (Status: ${response.status})`);
                return;
            }

            const jsonRecived = await response.json();
            console.log('Respuesta del servidor recibida');

            if (jsonRecived && jsonRecived.message && Array.isArray(jsonRecived.recent_list)) {
                // guardo los path de manera local para tenerlos disponibles en la pantalla de upload y redirijo a la pantalla anterior
                setRecents(jsonRecived.recent_list);
                setIsLoggedIn(true);
                setUser(jsonRecived.user);
                navigation.reset({
                    index: 0,
                    routes: [ { name: 'Home' } ],
                });
                
            } else {
                Alert.alert('Error', 'Respuesta inesperada del servidor');
                console.error('Respuesta inválida del servidor:', jsonRecived);
            }
        } catch (error) {
            console.error(error);
            Alert.alert('Error de conexión', 'No se pudo conectar con el servidor.');
        }
    }

	return (
		<KeyboardAvoidingView
			behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
			style={styles.container}
		>
			<AppBackground>
				<ScrollView contentContainerStyle={[styles.overlay, styles.scrollContent]}>
					<View style={styles.logoContainer}>
						<View style={styles.logo}>
							<Text style={styles.logoText}>🌱</Text>
						</View>
						<Text style={styles.title}>Crear Cuenta</Text>
						<Text style={styles.subtitle}>Empieza a gestionar tu cultivo en minutos</Text>
					</View>

					<View style={styles.formContainer}>
						<View style={styles.inputGroup}>
							<Text style={styles.label}>Nombre completo</Text>
							<View style={styles.inputWrapper}>
						<TextInput
							style={styles.input}
							placeholder="Tu nombre"
							placeholderTextColor="#999"
							value={name}
							onChangeText={setName}
						/>
							</View>
						</View>

						<View style={styles.inputGroup}>
							<Text style={styles.label}>Correo electrónico</Text>
							<View style={styles.inputWrapper}>
						<TextInput
							style={styles.input}
							placeholder="ejemplo@habichuela.com"
							placeholderTextColor="#999"
							keyboardType="email-address"
							value={email}
							onChangeText={setEmail}
						/>
							</View>
						</View>

						<View style={styles.inputGroup}>
							<Text style={styles.label}>Contraseña</Text>
							<View style={styles.inputWrapper}>
						<TextInput
							style={styles.input}
							placeholder="••••••••"
							placeholderTextColor="#999"
							secureTextEntry
							value={password}
							onChangeText={setPassword}
						/>
							</View>
						</View>

						<TouchableOpacity style={styles.registerButton} onPress={() => handleRegister(name, email, password)} activeOpacity={0.85}>
							<Text style={styles.registerButtonText}>Crear cuenta →</Text>
						</TouchableOpacity>

						<TouchableOpacity
							style={styles.backToLoginButton}
							onPress={() => navigation.goBack()}
						>
							<Text style={styles.backToLoginText}>Ya tengo cuenta, ir a iniciar sesión</Text>
						</TouchableOpacity>
					</View>
				</ScrollView>
			</AppBackground>
		</KeyboardAvoidingView>
	);
};

const styles = StyleSheet.create({
	container: {
		flex: 1,
		backgroundColor: 'transparent',
	},
	overlay: {
			flex: 1,
			backgroundColor: 'rgba(255, 255, 255, 0.65)',
	},
	scrollContent: {
		flexGrow: 1,
		justifyContent: 'center',
		paddingHorizontal: 20,
		paddingVertical: 40,
	},
	logoContainer: {
		alignItems: 'center',
		marginBottom: 46,
	},
	logo: {
		width: 60,
		height: 60,
		borderRadius: 30,
		backgroundColor: '#1B9A6E',
		justifyContent: 'center',
		alignItems: 'center',
		marginBottom: 20,
	},
	logoText: {
		fontSize: 30,
		color: '#fff',
	},
	title: {
		fontSize: 22,
		fontWeight: '700',
		color: '#333',
		marginBottom: 6,
		letterSpacing: 0.5,
	},
	subtitle: {
		fontSize: 12,
		color: '#888',
	},
	formContainer: {
		gap: 18,
	},
	inputGroup: {
		gap: 8,
	},
	label: {
		fontSize: 12,
		fontWeight: '600',
		color: '#333',
	},
	inputWrapper: {
		backgroundColor: '#fff',
		borderWidth: 1,
		borderColor: '#e0e0e0',
		borderRadius: 8,
		paddingHorizontal: 12,
		height: 48,
		justifyContent: 'center',
	},
	input: {
		fontSize: 14,
		color: '#333',
		paddingVertical: 12,
	},
	registerButton: {
		marginTop: 8,
		backgroundColor: '#1B9A6E',
		paddingVertical: 14,
		borderRadius: 8,
		alignItems: 'center',
	},
	registerButtonText: {
		color: '#fff',
		fontSize: 16,
		fontWeight: '600',
	},
	backToLoginButton: {
		alignItems: 'center',
		paddingVertical: 12,
	},
	backToLoginText: {
		color: '#167A57',
		fontSize: 13,
		fontWeight: '600',
	},
});

export default RegisterScreen;
