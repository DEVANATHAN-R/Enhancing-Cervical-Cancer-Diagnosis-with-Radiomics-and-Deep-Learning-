import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import SimpleITK as sitk
from radiomics import featureextractor, getFeatureClasses
import cv2

# Updated Constants
IMG_SIZE = (300, 300)  # Updated image size
BATCH_SIZE = 32
DATASET_PATH = "/content/drive/Othercomputers/My Laptop/FINAL YEAR PROJECT/dataset images"
METADATA_XLSX_PATH = "/content/drive/Othercomputers/My Laptop/FINAL YEAR PROJECT/dataset images/Updated_Cases_MetaData.xlsx"

def build_unet(input_shape):
    """Build U-Net model for image segmentation"""
    inputs = Input(input_shape)

    # Contracting Path (Encoder)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottom
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)

    # Expanding Path (Decoder)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    # Resize or crop `up5` to match `conv3`
    up5 = tf.keras.layers.Resizing(75, 75)(up5)  # Match conv3 dimensions
    merge5 = concatenate([conv3, up5], axis=3)

    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    # Resize or crop `up6` to match `conv2`
    up6 = tf.keras.layers.Resizing(150, 150)(up6)  # Match conv2 dimensions
    merge6 = concatenate([conv2, up6], axis=3)

    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    # Resize or crop `up7` to match `conv1`
    up7 = tf.keras.layers.Resizing(300, 300)(up7)  # Match conv1 dimensions
    merge7 = concatenate([conv1, up7], axis=3)

    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)


    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class RadiomicsExtractor:
    def __init__(self):
        self.settings = {
            'binWidth': 10,
            'normalize': True,
            'normalizeScale': 100,
            'force2D': True,
            'force2Ddimension': 0,
            'label': 255,
            'interpolator': 'sitkBSpline',
            'resampledPixelSpacing': [0.5, 0.5],
            'padDistance': 5,
            'distances': [1, 2, 3],
            'minimumROISize': 20,
            'additionalInfo': True
        }
        self._initialize_extractor()
        self.unet = build_unet((IMG_SIZE[0], IMG_SIZE[1], 1))

    def _initialize_extractor(self):
        """Initialize the radiomic feature extractor with proper settings"""
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.settings)
        feature_classes = getFeatureClasses()
        for feature_class in feature_classes:
            if feature_class != 'shape':
                self.extractor.enableFeatureClassByName(feature_class)
        self.extractor.enableFeatureClassByName('shape2D')
        self.extractor.enableImageTypeByName('Wavelet', {})

    def _square_crop(self, img):
        """Crop the image to a square"""
        height, width = img.shape[:2]
        size = min(height, width)
        start_y = (height - size) // 2
        start_x = (width - size) // 2
        cropped = img[start_y:start_y+size, start_x:start_x+size]
        return cropped

    def _preprocess_image(self, img):
        """Preprocess a single image"""
        img = self._square_crop(img)
        img = cv2.resize(img, IMG_SIZE)
        return cv2.equalizeHist(img)

    def _create_mask_unet(self, img):
        """Create mask using U-Net"""
        # Prepare image for U-Net
        input_img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0

        # Predict mask
        predicted_mask = self.unet.predict(input_img)[0]

        # Post-process the mask
        binary = (predicted_mask > 0.5).astype(np.uint8) * 255
        binary = binary.squeeze()

        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if np.sum(binary) == 0:
            # Fallback to traditional method if U-Net fails
            return self._create_mask_traditional(img)

        return binary

    def _create_mask_traditional(self, img):
        """Traditional mask creation as fallback"""
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if np.sum(binary) == 0:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary

    def extract_features(self, image_path):
        """Extract features from a single image"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            img = self._preprocess_image(img)
            binary = self._create_mask_unet(img)

            sitk_image = sitk.GetImageFromArray(img.astype(np.uint8))
            sitk_image.SetSpacing([0.5, 0.5])
            mask_sitk = sitk.GetImageFromArray(binary.astype(np.uint8))
            mask_sitk.SetSpacing([0.5, 0.5])

            features = self.extractor.execute(sitk_image, mask_sitk)
            features_dict = {key: value for key, value in features.items()
                           if isinstance(value, (int, float)) and not key.startswith('diagnostics')}

            return np.array(list(features_dict.values()))

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return np.zeros(93)

def preprocess_image(image_path):
    """Load and preprocess image for EfficientNet"""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return img_array

def load_metadata():
    """Load and process the metadata from Excel file with new target variables."""
    try:
        metadata = pd.read_excel(METADATA_XLSX_PATH)
        metadata.columns = metadata.columns.str.strip()

        label_encoder = LabelEncoder()
        metadata['severity_level_encoded'] = label_encoder.fit_transform(metadata['severity_level'])
        metadata['cancer_risk_encoded'] = label_encoder.fit_transform(metadata['cancer_risk'])

        if 'hpv_positive' not in metadata.columns:
            metadata['hpv_positive'] = metadata['HPV'].apply(
                lambda x: 1 if str(x).lower() == 'positive' else 0
            )

        encoded_columns = []

        if 'Type' in metadata.columns:
            encoder = OneHotEncoder(sparse_output=False)
            metadata_encoded = encoder.fit_transform(metadata[["Type"]])
            encoded_columns = encoder.get_feature_names_out(["Type"]).tolist()
            metadata[encoded_columns] = metadata_encoded

        case_to_images = {}
        for case_num in metadata['Case Number'].unique():
            case_folder = f"Case {case_num:03d}"
            image_files = sorted([f for f in os.listdir(os.path.join(DATASET_PATH, case_folder))
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:4]
            case_to_images[case_num] = image_files

        target_columns = ['severity_level_encoded', 'cancer_risk_encoded', 'hpv_positive']
        columns_to_group = encoded_columns + target_columns
        case_to_metadata = metadata.groupby("Case Number")[columns_to_group].first()

        return metadata, case_to_images, case_to_metadata, encoded_columns, target_columns

    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        raise

def load_case_data(case_number, case_to_images, case_to_metadata, encoded_columns, target_columns, radiomics_extractor):
    """Load case data with target variables."""
    folder_name = f"Case {case_number:03d}"
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        folder_name = f"case {case_number:03d}"
        folder_path = os.path.join(DATASET_PATH, folder_name)

    try:
        images = []
        radiomic_features = []

        for file_name in case_to_images[case_number]:
            image_path = os.path.join(folder_path, file_name)

            # Use the new RadiomicsExtractor class
            features = radiomics_extractor.extract_features(image_path)
            if features is not None and len(features) > 0:
                radiomic_features.append(features)

            try:
                img = preprocess_image(image_path)
                images.append(img)
            except Exception as e:
                print(f"Error preprocessing image {image_path}: {str(e)}")
                continue

        if not images or not radiomic_features:
            print(f"No valid data for case {case_number}")
            return None, None, None, None

        images = np.array(images)
        radiomic_features = np.array(radiomic_features)

        try:
            metadata_vector = case_to_metadata.loc[case_number, encoded_columns].values
            target_vector = case_to_metadata.loc[case_number, target_columns].values
            metadata_vector = np.array(metadata_vector, dtype=np.float32)
            target_vector = np.array(target_vector, dtype=np.float32)
        except Exception as e:
            print(f"Error processing metadata for case {case_number}: {str(e)}")
            metadata_vector = np.zeros(len(encoded_columns), dtype=np.float32)
            target_vector = np.zeros(len(target_columns), dtype=np.float32)

        return images, metadata_vector, radiomic_features, target_vector

    except Exception as e:
        print(f"Error processing case {case_number}: {str(e)}")
        return None, None, None, None

def main():
    # Initialize the RadiomicsExtractor
    radiomics_extractor = RadiomicsExtractor()

    # Load metadata
    metadata, case_to_images, case_to_metadata, encoded_columns, target_columns = load_metadata()

    # Process all cases
    all_data = []

    for case_number in case_to_images.keys():
        result = load_case_data(case_number, case_to_images, case_to_metadata,
                              encoded_columns, target_columns, radiomics_extractor)
        if result is not None:
            images, metadata_vector, radiomics_features, target_vector = result
            if all(x is not None for x in [images, metadata_vector, radiomics_features, target_vector]):
                # Process CNN features
                base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
                cnn_features = base_model.predict(images, batch_size=BATCH_SIZE)
                cnn_features_mean = np.mean(cnn_features, axis=0)

                # Average radiomic features
                radiomic_features_mean = np.mean(radiomics_features, axis=0)

                # Combine all features for this case
                case_data = {
                    'case_number': case_number,
                    **{f'cnn_feature_{i}': val for i, val in enumerate(cnn_features_mean)},
                    **{f'radiomic_feature_{i}': val for i, val in enumerate(radiomic_features_mean)},
                    **{f'metadata_{i}': val for i, val in enumerate(metadata_vector)},
                    **{f'target_{i}': val for i, val in enumerate(target_vector)}
                }
                all_data.append(case_data)

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(all_data)
    results_df.to_csv('/content/drive/Othercomputers/My Laptop/FINAL YEAR PROJECT/dataset images/combined_features.csv', index=False)

    print(f"\nProcessed {len(results_df)} cases successfully")
    print(f"Total features per case: {len(results_df.columns) - 1}")  # -1 for case_number

    # Print feature distributions
    feature_counts = {
        'CNN Features': len([col for col in results_df.columns if col.startswith('cnn_feature_')]),
        'Radiomic Features': len([col for col in results_df.columns if col.startswith('radiomic_feature_')]),
        'Metadata Features': len([col for col in results_df.columns if col.startswith('metadata_')]),
        'Target Variables': len([col for col in results_df.columns if col.startswith('target_')])
    }

    print("\nFeature Distribution:")
    for feature_type, count in feature_counts.items():
        print(f"{feature_type}: {count}")

    return results_df

if __name__ == "__main__":
    results_df = main()

