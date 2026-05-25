import React from 'react';
import { ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { RouteProp, useNavigation, useRoute } from '@react-navigation/native';
import { User } from '../classes/User';
import { useSession } from '../SessionContext';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import AppBackground from '../component/AppBackground';
import { Ionicons } from '@expo/vector-icons';

type RootStackParamList = {
    Home: undefined;
    Login: undefined;
    Register: undefined;
    Profile: { user?: User };
};

type ProfileScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>;


export default function Profile() {
	const navigation = useNavigation<ProfileScreenNavigationProp>();
    const { user, recents, setUser, setRecents, setIsLoggedIn } = useSession();

    function handleLogout() {
        setUser(null);
        setRecents([]);
        setIsLoggedIn(false);
        navigation.reset({
            index: 0,
            routes: [ { name: 'Home' } ],
        });
    }

	return (
        <AppBackground>
            <View style={[styles.container, styles.overlay]}>
                {user ? (
                <>
                    <View style={styles.logoContainer}>
                        <View style={styles.logo}>
                            <Ionicons name="person-outline" size={36} color="#fff" />
                        </View>
                        <Text style={styles.title}>Usuario</Text>
                        <Text style={styles.subtitle}>Informacion relevante para el usuario</Text>
                    </View>
                    <View>
                        <View style={styles.card}>
                            <Text style={styles.row}>
                                <Text style={styles.label}>Nombre: </Text>
                                <Text style={styles.value}>{user.username}</Text>
                            </Text>
                            <Text style={styles.row}>
                                <Text style={styles.label}>Email: </Text>
                                <Text style={styles.value}>{user.email}</Text>
                            </Text>
                            <Text style={styles.row}>
                                <Text style={styles.label}>Imágenes subidas: </Text>
                                <Text style={styles.value}>{recents.length}</Text>
                            </Text>
                            <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
                                <Text style={styles.logoutButtonText}>
                                    Cerrar Sesión
                                </Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </>
                ) : (
                    <View style={styles.card}>
                        <Text style={styles.empty}>
                            No se ha recibido ningún usuario. Pasa el objeto User en los parámetros de la ruta.
                        </Text>
                    </View>
                )}
            </View>
        </AppBackground>
	);
}

const styles = StyleSheet.create({
	container: {
		flexGrow: 1,
		padding: 24,
	},
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(255, 255, 255, 0.65)',
    },
	title: {
		fontSize: 28,
		fontWeight: '700',
		color: '#111827',
		marginBottom: 8,
	},
	subtitle: {
		fontSize: 16,
		color: '#000000',
		marginBottom: 20,
	},
	card: {
		backgroundColor: '#FFFFFF',
		borderRadius: 16,
		padding: 16,
		shadowColor: '#000',
		shadowOpacity: 0.08,
		shadowRadius: 12,
		shadowOffset: { width: 0, height: 4 },
		elevation: 3,
	},
	row: {
		marginBottom: 14,
		paddingBottom: 14,
		borderBottomWidth: 1,
		borderBottomColor: '#E5E7EB',
	},
	label: {
		fontSize: 13,
		color: '#6B7280',
		textTransform: 'capitalize',
		marginBottom: 4,
	},
	value: {
		fontSize: 16,
		color: '#111827',
		fontWeight: '500',
	},
	empty: {
		fontSize: 15,
		color: '#374151',
	},
    logoContainer: {
    alignItems: 'center',
    marginBottom: 50,
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
  logoutButton: {
    marginTop: 20,
    backgroundColor: '#1B9A6E',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  logoutButtonText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    }
});
