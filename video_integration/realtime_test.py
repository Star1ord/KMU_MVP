"""
ADVANCED REAL-TIME TESTING on webcam

BASIC FEATURES:
- Blinking (frequency, EAR)
- Emotions (7 basic + duration)
- Gestures (hand detection)

ADVANCED FEATURES:
- 3D Head Pose (Yaw, Pitch, Roll)
- Gaze Direction (Eye Gaze)
- Distance from Camera
- Head Down Position
- Face Touching with Hands
- Head Stability (tremor)
- Emotion Duration

Controls:
    Q - exit
    R - reset statistics
    E - toggle emotions on/off
    D - show/hide debug
    
Author: ML_Suicide Project v2.0
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Loading HSEmotion (better for Asian faces!)
print("\n=== HSEMOTION DIAGNOSTICS ===")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    
    # CUDA diagnostics
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  📊 CUDA version: {torch.version.cuda}")
        print(f"  💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  ⚠️ CUDA unavailable - using CPU")
    
    import timm
    print(f"✓ timm {timm.__version__}")
    if timm.__version__.startswith('0.9'):
        print("  ✅ Version compatible")
    elif not timm.__version__.startswith('0.6'):
        print(f"  ⚠️ WARNING: timm {timm.__version__} may be incompatible!")
        print(f"     Recommended: pip install timm==0.9.16")
    
    # FIX: PyTorch 2.6+ changed weights_only to True by default
    # HSEmotion requires weights_only=False to load the model
    # add_safe_globals method is only available in PyTorch 2.6+
    try:
        torch.serialization.add_safe_globals([
            'timm.models.efficientnet.EfficientNet',
            'timm.models.efficientnet.EfficientNetFeatures'
        ])
    except AttributeError:
        # Method unavailable in older PyTorch versions - this is normal
        pass
    
    print("  → Importing HSEmotionRecognizer...")
    from hsemotion.facial_emotions import HSEmotionRecognizer
    HSEMOTION_AVAILABLE = True
    print("✓ HSEmotion loaded successfully!")
except ImportError as e:
    HSEMOTION_AVAILABLE = False
    print("✗ HSEmotion not installed")
    print(f"  Import error: {e}")
    print("  Install: pip install hsemotion timm==0.6.13")
except Exception as e:
    HSEMOTION_AVAILABLE = False
    print("✗ HSEmotion unavailable")
    print(f"  Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
print("="*30 + "\n")


class RealtimeAnalyzer:
    """Advanced analyzer for webcam processing"""
    
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye indices for EAR and Iris
        self.LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]
        self.EAR_THRESHOLD = 0.21
        
        # Basic statistics
        self.blink_count = 0
        self.prev_ear = 1.0
        self.start_time = time.time()
        self.ear_history = deque(maxlen=30)
        
        # Emotions (HSEmotion - better for Asians)
        self.use_emotions = HSEMOTION_AVAILABLE
        self.emotion_model = None
        if self.use_emotions:
            try:
                # FIX for PyTorch 2.6: add safe globals
                import torch
                import warnings
                
                # Suppress PyTorch warnings
                warnings.filterwarnings('ignore', category=FutureWarning)
                
                # Add EfficientNet classes to safe globals
                try:
                    torch.serialization.add_safe_globals([
                        'timm.models.efficientnet.EfficientNet',
                        'timm.models.efficientnet.EfficientNetFeatures'
                    ])
                except:
                    pass
                
                # Alternative: patch torch.load
                old_load = torch.load
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return old_load(*args, **kwargs)
                torch.load = safe_load
                
                # Determine device: use CUDA if available, otherwise CPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                print(f"  [Loading HSEmotion model on {device.upper()}...]")
                self.emotion_model = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
                
                # Restore original torch.load
                torch.load = old_load
                
                # Check model device
                if hasattr(self.emotion_model, 'device'):
                    device_str = str(self.emotion_model.device)
                elif hasattr(self.emotion_model, 'model') and hasattr(self.emotion_model.model, 'parameters'):
                    try:
                        device_str = str(next(self.emotion_model.model.parameters()).device)
                    except:
                        device_str = device
                else:
                    device_str = device
                
                print("  ✓ HSEmotion model loaded successfully!")
                print(f"  📍 Model device: {device_str}")
                
                # Additional CUDA check
                if device == 'cuda' and device_str == 'cpu':
                    print("  ⚠️ WARNING: Model loaded on CPU, although CUDA is available!")
                    print("     Model may not support GPU or an error occurred.")
                elif device == 'cuda' and 'cuda' in device_str.lower():
                    # Verify that model is actually on GPU
                    try:
                        import torch
                        if hasattr(self.emotion_model, 'model'):
                            model_device = next(self.emotion_model.model.parameters()).device
                            if model_device.type == 'cuda':
                                print(f"  ✅ Confirmed: inference running on GPU ({torch.cuda.get_device_name(model_device.index)})")
                            else:
                                print(f"  ⚠️ Model on {model_device}, but CUDA was expected")
                    except:
                        pass
            except Exception as e:
                print(f"  ✗ Error loading HSEmotion model:")
                print(f"     {str(e)[:100]}")
                self.use_emotions = False
                print("  → Emotions will be DISABLED")
        
        self.last_emotion_time = 0
        self.emotion_interval = 2.0  # analyze every 2 seconds
        self.current_emotion = "unknown"
        self.emotion_scores = {}
        self.emotion_start_time = {}
        self.emotion_duration = 0.0
        
        # Hands
        self.hands_detected = 0
        
        # === ADVANCED FEATURES ===
        
        # 3D Head Pose
        self.head_yaw = 0.0  # left-right (-90 to 90)
        self.head_pitch = 0.0  # up-down (-90 to 90)
        self.head_roll = 0.0  # tilt (-90 to 90)
        
        # Gaze Direction
        self.gaze_direction = "center"  # left, right, up, down, center
        self.looking_at_camera = True
        self.gaze_away_time = 0.0  # seconds looking away from camera
        
        # Distance from Camera
        self.face_distance = 0.0  # relative (0-1)
        self.face_distance_history = deque(maxlen=30)
        self.baseline_distance = None
        
        # Head Down
        self.head_down = False
        self.head_down_duration = 0.0
        
        # Face Touching
        self.touching_face = False
        self.face_touch_count = 0
        
        # Head Stability (tremor)
        self.head_positions = deque(maxlen=10)  # last 10 positions
        self.head_stability = 1.0  # 0 = shaky, 1 = stable
        
        # Visual effects
        self.emotion_glow_alpha = 0.0  # for emotion glow effect
        self.face_bbox = None  # face bounding box for glow
        self.last_blink_time = 0.0  # for blink animation
        self.emotion_change_time = 0.0  # for emotion change animation
        self.prev_emotion = "unknown"  # previous emotion for transitions
        
        # Debug
        self.show_debug = False
        
        # Detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("\n" + "="*70)
        print(" 🚀 ADVANCED ANALYZER STARTED")
        print("="*70)
        
        # Device information for emotions
        if self.use_emotions and self.emotion_model:
            try:
                import torch
                if hasattr(self.emotion_model, 'model'):
                    model_device = next(self.emotion_model.model.parameters()).device
                    device_info = f"GPU ({torch.cuda.get_device_name(model_device.index)})" if model_device.type == 'cuda' else "CPU"
                    print(f"  💻 HSEmotion running on: {device_info}")
            except:
                pass
        
        print("Basic features:")
        print("  ✓ Blinking (frequency, EAR)")
        emotion_status = "HSEmotion (for Asians)" if self.use_emotions else "OFF"
        print(f"  ✓ Emotions (7 types + duration) [{emotion_status}]")
        print("  ✓ Hand gestures")
        print("\nAdvanced features:")
        print("  ✓ 3D Head Pose (Yaw, Pitch, Roll)")
        print("  ✓ Gaze Direction")
        print("  ✓ Distance from Camera")
        print("  ✓ Head Down")
        print("  ✓ Face Touching")
        print("  ✓ Head Stability")
        print("\nHotkeys:")
        print("  Q - Exit")
        print("  R - Reset statistics")
        print("  E - Toggle emotions on/off")
        print("  D - Show/hide debug")
        print("="*70 + "\n")
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        v1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        h = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h)
    
    def analyze_emotions(self, frame, face_landmarks=None):
        """
        Analyze emotions using HSEmotion
        Optimized for Asian faces!
        """
        if not self.use_emotions or self.emotion_model is None:
            return
        
        current_time = time.time()
        
        # Analyze no more than once every N seconds
        if current_time - self.last_emotion_time < self.emotion_interval:
            # Update current emotion duration
            if self.current_emotion in self.emotion_start_time:
                self.emotion_duration = current_time - self.emotion_start_time[self.current_emotion]
            return
        
        try:
            # HSEmotion works with BGR image directly
            # Model will find face itself, but better if we pass crop
            
            h, w = frame.shape[:2]
            face_img = frame  # can pass entire frame
            
            # If landmarks available, crop face for better accuracy
            if face_landmarks is not None:
                # Find face boundaries
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min = max(0, int(min(x_coords)) - 20)
                x_max = min(w, int(max(x_coords)) + 20)
                y_min = max(0, int(min(y_coords)) - 20)
                y_max = min(h, int(max(y_coords)) + 20)
                
                face_img = frame[y_min:y_max, x_min:x_max]
            
            # Check that face is not empty
            if face_img.size == 0 or face_img.shape[0] < 50 or face_img.shape[1] < 50:
                return
            
            # HSEmotion emotion analysis
            # Returns: (emotion_name, array[8]) - tuple of name and probabilities
            result = self.emotion_model.predict_emotions(face_img, logits=False)
            
            # HSEmotion returns tuple: (emotion_name, probabilities_array)
            if isinstance(result, tuple) and len(result) == 2:
                emotion_name_str, emotion_scores_raw = result
            else:
                # Fallback: if just array returned
                emotion_scores_raw = result
            
            # Convert numpy array to list
            try:
                emotion_list = list(emotion_scores_raw)
                if len(emotion_list) != 8:
                    print(f"✗ HSEmotion returned {len(emotion_list)} emotions instead of 8")
                    return
            except Exception as e:
                print(f"✗ Error converting HSEmotion result: {e}")
                return
            
            # Mapping to our format (compatibility with DeepFace)
            # HSEmotion: [Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise]
            emotions = {
                'angry': float(emotion_list[0] * 100),
                'disgust': float((emotion_list[2] + emotion_list[1]) * 50),  # Disgust + Contempt
                'fear': float(emotion_list[3] * 100),
                'happy': float(emotion_list[4] * 100),
                'neutral': float(emotion_list[5] * 100),
                'sad': float(emotion_list[6] * 100),
                'surprise': float(emotion_list[7] * 100)
            }
            
            # Find dominant emotion
            new_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Track emotion change
            if new_emotion != self.current_emotion:
                self.prev_emotion = self.current_emotion
                self.emotion_start_time[new_emotion] = current_time
                self.emotion_duration = 0.0
                self.emotion_change_time = current_time
                print(f"→ Emotion: {new_emotion.upper()} ({emotions[new_emotion]:.1f}%)")
            
            self.current_emotion = new_emotion
            self.emotion_scores = emotions
            self.last_emotion_time = current_time
            
        except Exception as e:
            # Show errors only in first 5 seconds for debugging
            if time.time() - self.start_time < 5:
                print(f"✗ Emotion analysis error: {type(e).__name__}: {str(e)[:100]}")
            # After 5 seconds silently ignore (to avoid cluttering console)
    
    def calculate_head_pose_3d(self, face_landmarks, frame_shape):
        """
        Calculate 3D head angles (Yaw, Pitch, Roll)
        USING READY 3D COORDINATES FROM MEDIAPIPE!
        """
        try:
            # MediaPipe provides 3D coordinates directly!
            # x, y - normalized (0-1)
            # z - relative depth (relative to nose)
            
            # Key points for angle calculation
            nose = face_landmarks.landmark[1]  # nose tip
            forehead = face_landmarks.landmark[10]  # forehead
            chin = face_landmarks.landmark[152]  # chin
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]
            
            # === YAW (left-right rotation) ===
            # Look at Z coordinate difference between left and right side of face
            # If left ear is farther (z larger) = turn right
            yaw_z_diff = right_ear.z - left_ear.z
            # Normalize (typical range -0.1 to 0.1)
            yaw = np.clip(yaw_z_diff * 500, -90, 90)  # scale to degrees
            
            # Also look at eyes
            eye_yaw = (right_eye.z - left_eye.z) * 300
            yaw = (yaw + eye_yaw) / 2  # average
            
            # === PITCH (up-down tilt) ===
            # Look at vertical position of nose relative to forehead and chin
            # and at Z coordinate of nose relative to forehead
            nose_y_relative = (nose.y - forehead.y) / (chin.y - forehead.y) if (chin.y - forehead.y) > 0 else 0.5
            
            # Z coordinate: if nose is higher than forehead in Z = head tilted down
            pitch_z_diff = forehead.z - nose.z
            pitch = np.clip(pitch_z_diff * 300, -90, 90)
            
            # Correction by Y position
            if nose_y_relative < 0.4:  # nose high = head up
                pitch -= 20
            elif nose_y_relative > 0.6:  # nose low = head down
                pitch += 20
            
            # === ROLL (side tilt) ===
            # Look at Y coordinate difference between eyes
            roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x) * 180 / np.pi
            
            # === SMOOTHING (Exponential Moving Average) ===
            alpha = 0.3  # smoothing coefficient (0 = no smoothing, 1 = only current)
            self.head_yaw = alpha * yaw + (1 - alpha) * self.head_yaw
            self.head_pitch = alpha * pitch + (1 - alpha) * self.head_pitch
            self.head_roll = alpha * roll + (1 - alpha) * self.head_roll
            
            # Determine head down
            self.head_down = self.head_pitch < -15
            
            return True, None  # return success, but without rotation_vec
            
        except Exception as e:
            # If something went wrong, keep previous values
            return None, None
    
    def calculate_gaze_direction(self, face_landmarks, frame_shape):
        """
        Determine gaze direction via iris tracking
        """
        h, w = frame_shape[:2]
        
        try:
            # Eye centers
            left_eye_center = np.mean([(face_landmarks.landmark[i].x * w,
                                        face_landmarks.landmark[i].y * h)
                                       for i in self.LEFT_EYE_IDXS], axis=0)
            right_eye_center = np.mean([(face_landmarks.landmark[i].x * w,
                                         face_landmarks.landmark[i].y * h)
                                        for i in self.RIGHT_EYE_IDXS], axis=0)
            
            # Iris centers (if available)
            if len(face_landmarks.landmark) > 468:
                left_iris_center = np.mean([(face_landmarks.landmark[i].x * w,
                                             face_landmarks.landmark[i].y * h)
                                            for i in self.LEFT_IRIS], axis=0)
                right_iris_center = np.mean([(face_landmarks.landmark[i].x * w,
                                              face_landmarks.landmark[i].y * h)
                                             for i in self.RIGHT_IRIS], axis=0)
                
                # Iris offset relative to eye center
                left_offset = left_iris_center - left_eye_center
                right_offset = right_iris_center - right_eye_center
                avg_offset = (left_offset + right_offset) / 2
                
                # Determine direction
                threshold_x = 5
                threshold_y = 5
                
                if abs(avg_offset[0]) < threshold_x and abs(avg_offset[1]) < threshold_y:
                    self.gaze_direction = "center"
                    self.looking_at_camera = True
                elif avg_offset[0] > threshold_x:
                    self.gaze_direction = "right"
                    self.looking_at_camera = False
                elif avg_offset[0] < -threshold_x:
                    self.gaze_direction = "left"
                    self.looking_at_camera = False
                elif avg_offset[1] > threshold_y:
                    self.gaze_direction = "down"
                    self.looking_at_camera = False
                elif avg_offset[1] < -threshold_y:
                    self.gaze_direction = "up"
                    self.looking_at_camera = False
                
                return left_iris_center, right_iris_center
        except:
            pass
        
        return None, None
    
    def calculate_face_distance(self, face_landmarks, frame_shape):
        """
        Calculate relative distance from camera
        Using distance between eyes
        """
        h, w = frame_shape[:2]
        
        # Distance between eyes in pixels
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        eye_distance_px = distance.euclidean(
            (left_eye.x * w, left_eye.y * h),
            (right_eye.x * w, right_eye.y * h)
        )
        
        # Normalization (average distance ~100-150 pixels)
        self.face_distance = eye_distance_px
        self.face_distance_history.append(eye_distance_px)
        
        # Set baseline on first run
        if self.baseline_distance is None and len(self.face_distance_history) > 10:
            self.baseline_distance = np.mean(self.face_distance_history)
        
        return eye_distance_px
    
    def check_hand_face_contact(self, face_landmarks, hands_landmarks, frame_shape):
        """
        Check if hand touches face
        """
        if not hands_landmarks or not face_landmarks:
            self.touching_face = False
            return False
        
        h, w = frame_shape[:2]
        
        # Face boundaries (approximate)
        face_points = []
        for idx in [10, 234, 454, 152]:  # forehead, left cheek, right cheek, chin
            landmark = face_landmarks.landmark[idx]
            face_points.append([landmark.x * w, landmark.y * h])
        
        face_points = np.array(face_points)
        face_center = np.mean(face_points, axis=0)
        face_radius = np.max([distance.euclidean(face_center, p) for p in face_points]) * 1.2
        
        # Check hand proximity to face
        for hand_landmarks in hands_landmarks:
            for landmark in hand_landmarks.landmark:
                hand_point = np.array([landmark.x * w, landmark.y * h])
                dist = distance.euclidean(hand_point, face_center)
                
                if dist < face_radius:
                    if not self.touching_face:
                        self.face_touch_count += 1
                    self.touching_face = True
                    return True
        
        self.touching_face = False
        return False
    
    def calculate_head_stability(self):
        """
        Calculate head stability (inverse tremor)
        """
        if len(self.head_positions) < 5:
            return
        
        # Position variance
        positions = np.array(self.head_positions)
        variance = np.var(positions, axis=0).mean()
        
        # Normalization (0 = many movements, 1 = stable)
        # Typical variance: 0-0.01 (stable), 0.01-0.1 (normal), >0.1 (shaky)
        self.head_stability = max(0.0, min(1.0, 1.0 - (variance * 100)))
    
    def process_frame(self, frame):
        """Process single frame with ALL features"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process detectors
        face_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        
        avg_ear = None
        
        # === FACE AND BLINKING ===
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # EAR for eyes
            left_eye_coords = np.array([(face_landmarks.landmark[i].x * w,
                                         face_landmarks.landmark[i].y * h)
                                        for i in self.LEFT_EYE_IDXS])
            right_eye_coords = np.array([(face_landmarks.landmark[i].x * w,
                                          face_landmarks.landmark[i].y * h)
                                         for i in self.RIGHT_EYE_IDXS])
            
            left_ear = self.calculate_ear(left_eye_coords)
            right_ear = self.calculate_ear(right_eye_coords)
            avg_ear = (left_ear + right_ear) / 2
            
            self.ear_history.append(avg_ear)
            
            # Blink detection
            if avg_ear < self.EAR_THRESHOLD and self.prev_ear >= self.EAR_THRESHOLD:
                self.blink_count += 1
                self.last_blink_time = time.time()
            
            self.prev_ear = avg_ear
            
            # === ADVANCED FEATURES ===
            
            # 1. 3D Head Pose (using ready MediaPipe 3D coordinates)
            self.calculate_head_pose_3d(face_landmarks, frame.shape)
            
            # 2. Gaze Direction
            left_iris, right_iris = self.calculate_gaze_direction(face_landmarks, frame.shape)
            
            # 3. Distance from Camera
            self.calculate_face_distance(face_landmarks, frame.shape)
            
            # 4. Head position for stability
            nose = face_landmarks.landmark[1]
            self.head_positions.append([nose.x, nose.y])
            self.calculate_head_stability()
            
            # Draw beautiful face visualization
            self.draw_face_visualization(frame, face_landmarks, left_eye_coords, right_eye_coords, left_iris, right_iris)
            
            # Draw head direction arrow (if debug)
            self.draw_head_axes_simple(frame, face_landmarks)
        
        # === HANDS ===
        hands_landmarks_list = []
        self.hands_detected = 0
        if hands_results.multi_hand_landmarks:
            self.hands_detected = len(hands_results.multi_hand_landmarks)
            hands_landmarks_list = hands_results.multi_hand_landmarks
            
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Beautiful hand visualization
                self.draw_hand_visualization(frame, hand_landmarks)
        
        # 5. Face Touching
        if face_results.multi_face_landmarks:
            self.check_hand_face_contact(
                face_results.multi_face_landmarks[0],
                hands_landmarks_list if hands_landmarks_list else None,
                frame.shape
            )
        
        # === EMOTION ANALYSIS (periodically) ===
        # Pass face_landmarks for more accurate face crop (for HSEmotion)
        face_landmarks_for_emotion = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        self.analyze_emotions(frame, face_landmarks_for_emotion)
        
        # Draw emotion visualization with glow effect
        if face_results.multi_face_landmarks:
            self.draw_emotion_visualization(frame, face_results.multi_face_landmarks[0])
        
        # Update head down duration
        if self.head_down:
            self.head_down_duration += 1/30  # approximately 30 FPS
        else:
            self.head_down_duration = 0
        
        # Update gaze away time
        if not self.looking_at_camera:
            self.gaze_away_time += 1/30
        else:
            self.gaze_away_time = max(0, self.gaze_away_time - 1/30)
        
        # === DRAW INFORMATION ===
        self.draw_info(frame, avg_ear)
        
        return frame
    
    def draw_face_visualization(self, frame, face_landmarks, left_eye_coords, right_eye_coords, left_iris, right_iris):
        """Draw simple face mesh grid with eyes and pupils"""
        h, w, _ = frame.shape
        
        # Get face bounding box for emotion visualization
        x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        self.face_bbox = (x_min, y_min, x_max, y_max)
        
        # Draw face mesh tesselation (grid) in neutral color with transparency
        face_tesselation = self.mp_face_mesh.FACEMESH_TESSELATION
        neutral_color = (150, 150, 150)  # Gray color
        
        # Create overlay for transparent mesh
        overlay = frame.copy()
        
        for connection in face_tesselation:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx < len(face_landmarks.landmark) and end_idx < len(face_landmarks.landmark):
                start = face_landmarks.landmark[start_idx]
                end = face_landmarks.landmark[end_idx]
                cv2.line(overlay,
                        (int(start.x * w), int(start.y * h)),
                        (int(end.x * w), int(end.y * h)),
                        neutral_color, 1, cv2.LINE_AA)
        
        # Draw eyes on overlay
        for coords in [left_eye_coords, right_eye_coords]:
            coords = coords.astype(int)
            cv2.polylines(overlay, [coords], True, neutral_color, 1, cv2.LINE_AA)
        
        # Blend overlay with original frame (0.6 = 60% overlay, 40% original = more transparent)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw irises/pupils ONLY if eyes are open (EAR > threshold)
        # Check if eyes are open by calculating EAR for both eyes
        left_ear = self.calculate_ear(left_eye_coords) if len(left_eye_coords) >= 6 else 1.0
        right_ear = self.calculate_ear(right_eye_coords) if len(right_eye_coords) >= 6 else 1.0
        
        # Only draw irises if both eyes are open
        if left_iris is not None and right_iris is not None:
            if left_ear > self.EAR_THRESHOLD and right_ear > self.EAR_THRESHOLD:
                for iris in [left_iris, right_iris]:
                    iris_int = iris.astype(int)
                    cv2.circle(frame, tuple(iris_int), 3, neutral_color, 1)
    
    def draw_emotion_visualization(self, frame, face_landmarks):
        """Draw minimal professional emotion indicator"""
        if not self.use_emotions or self.current_emotion == "unknown":
            return
        
        h, w, _ = frame.shape
        
        # Emotion color mapping - subtle colors
        emotion_colors = {
            'happy': (0, 200, 100),      # Green
            'sad': (100, 100, 255),       # Blue
            'angry': (0, 0, 200),         # Red
            'fear': (200, 0, 200),        # Magenta
            'neutral': (150, 150, 150),   # Gray
            'surprise': (0, 200, 255),    # Cyan
            'disgust': (0, 150, 100)      # Dark green
        }
        
        emotion_color = emotion_colors.get(self.current_emotion, (200, 200, 200))
        
        # Draw subtle emotion indicator in top-right corner (not on face)
        if self.face_bbox:
            x_min, y_min, x_max, y_max = self.face_bbox
            # Position indicator in top-right of screen, not overlapping face
            indicator_x = min(w - 120, x_max + 20) if x_max < w - 120 else w - 120
            indicator_y = max(20, y_min - 40) if y_min > 40 else 20
            
            # Minimal emotion badge
            badge_w, badge_h = 100, 35
            cv2.rectangle(frame, 
                        (indicator_x, indicator_y),
                        (indicator_x + badge_w, indicator_y + badge_h),
                        emotion_color, -1)
            cv2.rectangle(frame,
                        (indicator_x, indicator_y),
                        (indicator_x + badge_w, indicator_y + badge_h),
                        (255, 255, 255), 1)
            
            # Emotion text
            emotion_text = self.current_emotion.upper()
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = indicator_x + (badge_w - text_size[0]) // 2
            text_y = indicator_y + (badge_h + text_size[1]) // 2
            cv2.putText(frame, emotion_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Confidence percentage below badge
            if self.emotion_scores and self.current_emotion in self.emotion_scores:
                confidence = self.emotion_scores[self.current_emotion]
                conf_text = f"{confidence:.1f}%"
                conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                conf_x = indicator_x + (badge_w - conf_size[0]) // 2
                conf_y = indicator_y + badge_h + 15
                cv2.putText(frame, conf_text, (conf_x, conf_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_hand_visualization(self, frame, hand_landmarks):
        """Draw minimal professional hand visualization using MediaPipe standard style"""
        # Use MediaPipe's standard drawing - clean and professional
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 150, 200), thickness=1)
        )
    
    def draw_head_axes_simple(self, frame, face_landmarks):
        """Simple head direction visualization"""
        if not self.show_debug:
            return
        
        h, w, _ = frame.shape
        
        # Face center (nose)
        nose = face_landmarks.landmark[1]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        
        # Draw arrow showing head direction
        # Based on Yaw and Pitch
        
        # Arrow length
        arrow_length = 100
        
        # Calculate arrow end based on angles
        yaw_rad = self.head_yaw * np.pi / 180
        pitch_rad = self.head_pitch * np.pi / 180
        
        # Projection to 2D
        end_x = int(nose_x + arrow_length * np.sin(yaw_rad))
        end_y = int(nose_y - arrow_length * np.sin(pitch_rad))
        
        # Draw arrow
        cv2.arrowedLine(frame, (nose_x, nose_y), (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
        
        # Text with angles
        cv2.putText(frame, f"Y:{self.head_yaw:.0f} P:{self.head_pitch:.0f} R:{self.head_roll:.0f}",
                   (nose_x + 10, nose_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    def draw_info(self, frame, avg_ear):
        """Draw ALL information on frame"""
        h, w, _ = frame.shape
        
        # Enlarged semi-transparent panel
        overlay = frame.copy()
        panel_height = 320 if not self.show_debug else 450
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        blink_rate = (self.blink_count / elapsed_time * 60) if elapsed_time > 0 else 0
        
        # === LEFT COLUMN ===
        y = 25
        lh = 28  # line height
        
        # BLINKING
        cv2.putText(frame, "BLINKING:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += lh
        if avg_ear is not None:
            color = (0, 255, 0) if avg_ear > self.EAR_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if avg_ear < self.EAR_THRESHOLD:
                cv2.putText(frame, "[BLINK!]", (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        y += lh
        cv2.putText(frame, f"Total: {self.blink_count} | {blink_rate:.1f}/min", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 3D HEAD POSE ⭐
        y += lh + 5
        cv2.putText(frame, "3D HEAD POSE:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += lh
        cv2.putText(frame, f"Yaw:   {self.head_yaw:>6.1f}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        # Rotation indicator
        if abs(self.head_yaw) > 20:
            direction = "<- LEFT" if self.head_yaw < 0 else "RIGHT ->"
            cv2.putText(frame, direction, (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        y += lh
        pitch_color = (0, 0, 255) if self.head_pitch < -15 else (255, 255, 255)
        cv2.putText(frame, f"Pitch: {self.head_pitch:>6.1f}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_color, 1)
        if self.head_down:
            cv2.putText(frame, "[HEAD DOWN!]", (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        y += lh
        cv2.putText(frame, f"Roll:  {self.head_roll:>6.1f}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # GAZE DIRECTION ⭐
        y += lh + 5
        cv2.putText(frame, "GAZE:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += lh
        gaze_color = (0, 255, 0) if self.looking_at_camera else (255, 100, 0)
        gaze_text = "AT CAMERA" if self.looking_at_camera else f"AWAY ({self.gaze_direction})"
        cv2.putText(frame, gaze_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 2)
        
        if not self.looking_at_camera and self.gaze_away_time > 2:
            y += lh
            cv2.putText(frame, f"[Away {self.gaze_away_time:.1f}s]", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 50, 0), 1)
        
        # DISTANCE ⭐
        y += lh + 5
        if self.baseline_distance:
            distance_ratio = self.face_distance / self.baseline_distance
            dist_text = "CLOSE" if distance_ratio > 1.1 else ("FAR" if distance_ratio < 0.9 else "NORMAL")
            dist_color = (0, 255, 255) if 0.9 <= distance_ratio <= 1.1 else (255, 100, 0)
            cv2.putText(frame, f"Dist: {dist_text} ({distance_ratio:.2f}x)", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
            y += lh
        
        # STABILITY ⭐
        stability_percent = self.head_stability * 100
        stab_color = (0, 255, 0) if stability_percent > 80 else ((255, 255, 0) if stability_percent > 50 else (0, 0, 255))
        cv2.putText(frame, f"Stability: {stability_percent:.0f}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, stab_color, 1)
        if stability_percent < 50:
            cv2.putText(frame, "[SHAKY!]", (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        
        # === RIGHT COLUMN ===
        x_right = w // 2 + 50
        y = 25
        
        # EMOTIONS
        cv2.putText(frame, "EMOTIONS:", (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += lh
        
        if self.use_emotions and self.current_emotion != "unknown":
            emotion_colors = {
                'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
                'fear': (255, 0, 255), 'neutral': (200, 200, 200), 'surprise': (0, 255, 255)
            }
            color = emotion_colors.get(self.current_emotion, (255, 255, 255))
            cv2.putText(frame, self.current_emotion.upper(), (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Emotion duration ⭐
            y += lh
            cv2.putText(frame, f"Duration: {self.emotion_duration:.1f}s", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if self.emotion_duration > 10 and self.current_emotion in ['sad', 'fear']:
                cv2.putText(frame, "[LONG!]", (x_right + 180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            
            # Top-2 emotions with bars
            if self.emotion_scores:
                y += lh - 5
                sorted_emotions = sorted(self.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                for emotion, score in sorted_emotions:
                    bar_w = int(score * 1.5)
                    cv2.rectangle(frame, (x_right, y - 12), (x_right + bar_w, y - 4),
                                 emotion_colors.get(emotion, (255, 255, 255)), -1)
                    cv2.putText(frame, f"{emotion}: {score:.0f}%", (x_right + 160, y - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    y += 18
        else:
            cv2.putText(frame, "OFF", (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            y += lh
        
        # HANDS AND FACE TOUCHING ⭐
        y += lh
        cv2.putText(frame, "HANDS:", (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
        y += lh
        
        hands_color = (0, 255, 0) if self.hands_detected > 0 else (128, 128, 128)
        cv2.putText(frame, f"In frame: {self.hands_detected}", (x_right, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hands_color, 1)
        
        y += lh
        if self.touching_face:
            cv2.putText(frame, "[TOUCHING FACE!]", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
        else:
            cv2.putText(frame, "Not touching", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        
        y += lh
        cv2.putText(frame, f"Total touches: {self.face_touch_count}", (x_right, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # ALERTS (CRITICAL MARKERS) ⭐
        y += lh + 10
        cv2.putText(frame, "RISK MARKERS:", (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        y += lh
        
        risk_count = 0
        if self.head_down and self.head_down_duration > 3:
            cv2.putText(frame, f"! Head down {self.head_down_duration:.0f}s", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y += lh - 5
            risk_count += 1
        
        if not self.looking_at_camera and self.gaze_away_time > 5:
            cv2.putText(frame, f"! Avoiding gaze {self.gaze_away_time:.0f}s", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y += lh - 5
            risk_count += 1
        
        if self.current_emotion == 'sad' and self.emotion_duration > 15:
            cv2.putText(frame, f"! Sadness {self.emotion_duration:.0f}s", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y += lh - 5
            risk_count += 1
        
        if self.head_stability < 0.5:
            cv2.putText(frame, "! Head tremor", (x_right, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y += lh - 5
            risk_count += 1
        
        if risk_count == 0:
            cv2.putText(frame, "None", (x_right, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        
        # === BOTTOM OF SCREEN ===
        # Time and controls
        cv2.putText(frame, f"Time: {minutes:02d}:{seconds:02d}", (10, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Q-exit | R-reset | E-emotions | D-debug", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Risk indicator (large!)
        if risk_count > 0:
            risk_text = f"RISK LEVEL: {risk_count}/4"
            risk_color = (0, 165, 255) if risk_count == 1 else ((0, 100, 255) if risk_count == 2 else (0, 0, 255))
            cv2.rectangle(frame, (w - 250, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, risk_text, (w - 240, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
    
    def reset_stats(self):
        """Сброс статистики"""
        self.blink_count = 0
        self.start_time = time.time()
        self.ear_history.clear()
        print("✓ Статистика сброшена")
    
    def toggle_emotions(self):
        """Переключение анализа эмоций (HSEmotion)"""
        if HSEMOTION_AVAILABLE and self.emotion_model is not None:
            self.use_emotions = not self.use_emotions
            status = "ВКЛЮЧЕН" if self.use_emotions else "ВЫКЛЮЧЕН"
            print(f"✓ Анализ эмоций {status} (HSEmotion)")
        else:
            print("✗ HSEmotion недоступен - установите: pip install hsemotion timm")
    
    def toggle_debug(self):
        """Переключение режима отладки"""
        self.show_debug = not self.show_debug
        status = "ВКЛЮЧЕН" if self.show_debug else "ВЫКЛЮЧЕН"
        print(f"✓ Режим отладки {status}")
    
    def cleanup(self):
        """Очистка ресурсов"""
        self.face_mesh.close()
        self.pose.close()
        self.hands.close()


def main():
    """Main function"""
    print("\n" + "="*60)
    print(" WEBCAM TESTING")
    print("="*60)
    print("\nStarting camera...\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Failed to open camera!")
        print("Check:")
        print("  1. Camera is connected")
        print("  2. Camera is not used by another application")
        print("  3. Camera drivers are installed")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✓ Camera opened successfully!")
    
    # Create analyzer
    analyzer = RealtimeAnalyzer()
    
    print("\n🎥 Press Q to exit\n")
    
    # Основной цикл
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Mirror for convenience
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = analyzer.process_frame(frame)
            
            # FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            cv2.putText(frame, f"FPS: {current_fps}", 
                       (frame.shape[1] - 100, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show result
            cv2.imshow('ML_Suicide - Realtime Test', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\n👋 Exiting...")
                break
            elif key == ord('r') or key == ord('R'):
                analyzer.reset_stats()
            elif key == ord('e') or key == ord('E'):
                analyzer.toggle_emotions()
            elif key == ord('d') or key == ord('D'):
                analyzer.toggle_debug()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        # Final statistics
        elapsed = time.time() - analyzer.start_time
        print("\n" + "="*70)
        print(" 📊 FINAL STATISTICS (ADVANCED)")
        print("="*70)
        
        print(f"\n⏱️  Session Duration: {int(elapsed // 60)}:{int(elapsed % 60):02d}")
        
        print(f"\n👁️  Blinking:")
        print(f"   Total: {analyzer.blink_count} times")
        print(f"   Rate: {(analyzer.blink_count / elapsed * 60):.1f} times/min")
        if analyzer.ear_history:
            print(f"   Average EAR: {np.mean(analyzer.ear_history):.3f}")
        
        print(f"\n😊 Emotions:")
        print(f"   Last: {analyzer.current_emotion}")
        print(f"   Duration: {analyzer.emotion_duration:.1f}s")
        
        print(f"\n🔄 Head Pose (final):")
        print(f"   Yaw:   {analyzer.head_yaw:>6.1f}°")
        print(f"   Pitch: {analyzer.head_pitch:>6.1f}°")
        print(f"   Roll:  {analyzer.head_roll:>6.1f}°")
        
        print(f"\n👀 Gaze:")
        print(f"   Direction: {analyzer.gaze_direction}")
        print(f"   Looking at camera: {'Yes ✓' if analyzer.looking_at_camera else 'No ✗'}")
        
        print(f"\n👐 Hands:")
        print(f"   Face touches: {analyzer.face_touch_count}")
        
        print(f"\n💎 Head Stability:")
        print(f"   {analyzer.head_stability * 100:.0f}%")
        
        print("\n⚠️  Detected Risk Markers:")
        risk_markers = []
        if analyzer.head_down and analyzer.head_down_duration > 3:
            risk_markers.append(f"   - Head down ({analyzer.head_down_duration:.0f}s)")
        if not analyzer.looking_at_camera and analyzer.gaze_away_time > 5:
            risk_markers.append(f"   - Avoiding gaze ({analyzer.gaze_away_time:.0f}s)")
        if analyzer.current_emotion == 'sad' and analyzer.emotion_duration > 15:
            risk_markers.append(f"   - Prolonged sadness ({analyzer.emotion_duration:.0f}s)")
        if analyzer.head_stability < 0.5:
            risk_markers.append(f"   - Head tremor ({analyzer.head_stability*100:.0f}%)")
        if analyzer.face_touch_count > 20:
            risk_markers.append(f"   - Frequent face touching ({analyzer.face_touch_count} times)")
        
        if risk_markers:
            for marker in risk_markers:
                print(marker)
            print(f"\n   🚨 RISK LEVEL: {len(risk_markers)}/5")
        else:
            print("   ✅ No risk markers detected")
        
        print("\n" + "="*70 + "\n")
        
        # Cleanup
        analyzer.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        
        print("✓ Resources released")
        print("Thank you for using the advanced analyzer! 👋\n")


if __name__ == "__main__":
    main()

