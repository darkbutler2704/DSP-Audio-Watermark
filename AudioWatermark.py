import streamlit as st
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import io
import base64

def text_to_binary(text):
    """Convert text to binary string with padding"""
    binary = ''.join(format(ord(char), '08b') for char in text)
    return binary

def binary_to_text(binary_str):
    """Convert binary string to text"""
    if len(binary_str) % 8 != 0:
        binary_str = binary_str[:len(binary_str) - (len(binary_str) % 8)]
    
    text = ""
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            try:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:
                    text += chr(char_code)
                elif char_code == 0:
                    break
            except ValueError:
                break
    return text

def embed_watermark_fft(audio_data, sample_rate, watermark_text, freq_start=2000, freq_width=1000, strength=0.3):
    """ Robust watermark embedding using consistent frequency domain manipulation with reference magnitudes """
    #Convert text to binary
    binary_watermark = text_to_binary(watermark_text)
    watermark_length = len(binary_watermark)
    
    #Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    #Convert to float64 for high precision processing
    audio_float = audio_data.astype(np.float64)
    
    #Normalize input audio to prevent overflow
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        audio_float = audio_float / max_val * 0.9
    
    #Apply FFT
    audio_fft = np.fft.fft(audio_float)
    frequencies = np.fft.fftfreq(len(audio_float), 1/sample_rate)
    
    #Calculate frequency bins more precisely
    freq_end = freq_start + freq_width
    freq_resolution = sample_rate / len(audio_float)
    
    #Find frequency bins in the positive half of spectrum
    start_bin = int(freq_start / freq_resolution)
    end_bin = int(freq_end / freq_resolution)
    
    #Ensure we have enough bins with spacing to avoid interference
    available_bins = list(range(start_bin, min(end_bin, len(audio_fft)//2)))
    
    if len(available_bins) < watermark_length * 3:
        raise ValueError(f"Insufficient frequency bins. Need {watermark_length * 3}, have {len(available_bins)}")
    
    step = max(3, len(available_bins) // watermark_length)
    selected_bins = available_bins[::step][:watermark_length]
    
    # Store original FFT and calculate reference magnitudes
    modified_fft = audio_fft.copy()
    reference_magnitudes = []
    
    # Calculate reference magnitudes for each selected bin using nearby unmodified bins
    for bin_idx in selected_bins:
        ref_bins = []
        for offset in [-4, -3, 3, 4]:
            ref_idx = bin_idx + offset
            if 0 <= ref_idx < len(audio_fft)//2 and ref_idx not in selected_bins:
                ref_bins.append(ref_idx)
        
        if ref_bins:
            ref_magnitude = np.mean([np.abs(audio_fft[idx]) for idx in ref_bins])
        else:
            ref_magnitude = np.abs(audio_fft[bin_idx])
        
        reference_magnitudes.append(ref_magnitude)
    
    #Embed watermark using robust amplitude modulation
    for i, bit in enumerate(binary_watermark):
        bin_idx = selected_bins[i]
        reference_mag = reference_magnitudes[i]
        phase = np.angle(modified_fft[bin_idx])
        if bit == '1':
            #Significantly amplify for bit 1
            new_magnitude = reference_mag * (2.0 + strength * 2.0)
        else:
            #Reduce magnitude for bit 0
            new_magnitude = reference_mag * (0.5 - strength * 0.4)
        
        #Ensure minimum magnitude and avoid clipping
        new_magnitude = max(new_magnitude, reference_mag * 0.05)
        new_magnitude = min(new_magnitude, reference_mag * 5.0)
        
        #Apply modification to positive frequency
        modified_fft[bin_idx] = new_magnitude * np.exp(1j * phase)
        
        #Apply to corresponding negative frequency for symmetry
        negative_bin = len(modified_fft) - bin_idx
        if negative_bin < len(modified_fft) and negative_bin != bin_idx:
            modified_fft[negative_bin] = np.conj(modified_fft[bin_idx])
    
    #Convert back to time domain
    watermarked_audio = np.fft.ifft(modified_fft).real
    
    max_output = np.max(np.abs(watermarked_audio))
    if max_output > 0:
        watermarked_audio = watermarked_audio / max_output * 0.95

    watermarked_audio = (watermarked_audio * 32767).astype(np.int16)
    
    #Store embedding parameters including reference magnitudes
    params = {
        'text': watermark_text,
        'binary': binary_watermark,
        'text_length': len(watermark_text),
        'binary_length': watermark_length,
        'freq_start': freq_start,
        'freq_width': freq_width,
        'strength': strength,
        'selected_bins': selected_bins,
        'reference_magnitudes': reference_magnitudes,
        'sample_rate': sample_rate,
        'freq_resolution': freq_resolution
    }
    
    return watermarked_audio, params

def extract_watermark_fft(audio_data, sample_rate, params):
    """ Parameter-independent watermark extraction that can work with any embedding parameters """
    # Extract basic parameters
    binary_length = params['binary_length']
    freq_start = params['freq_start']
    freq_width = params['freq_width']
    strength = params['strength']
    
    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
    
    # Convert to float64 for processing
    audio_float = audio_data.astype(np.float64)
    
    # Apply FFT
    audio_fft = np.fft.fft(audio_float)
    
    #Recalculate frequency bins independently
    freq_end = freq_start + freq_width
    freq_resolution = sample_rate / len(audio_float)
    start_bin = int(freq_start / freq_resolution)
    end_bin = int(freq_end / freq_resolution)
    
    # Reconstruct available bins
    available_bins = list(range(start_bin, min(end_bin, len(audio_fft)//2)))
    step = max(3, len(available_bins) // binary_length)
    selected_bins = available_bins[::step][:binary_length]
    
    # Extract bits using robust reference-based detection
    extracted_binary = ""
    confidence_scores = []
    
    for i, bin_idx in enumerate(selected_bins):
        if bin_idx < len(audio_fft):
            current_magnitude = np.abs(audio_fft[bin_idx])
            
            # Calculate reference from surrounding unmodified bins
            ref_bins = []
            for offset in [-6, -5, -4, 4, 5, 6]:  # Use wider range for robustness
                ref_idx = bin_idx + offset
                if 0 <= ref_idx < len(audio_fft)//2 and ref_idx not in selected_bins:
                    ref_bins.append(ref_idx)
            
            if ref_bins:
                reference_magnitude = np.median([np.abs(audio_fft[idx]) for idx in ref_bins])
            else:
                # Fallback: use local spectrum average
                start_idx = max(0, bin_idx - 10)
                end_idx = min(len(audio_fft)//2, bin_idx + 10)
                local_mags = [np.abs(audio_fft[idx]) for idx in range(start_idx, end_idx) 
                             if idx not in selected_bins]
                reference_magnitude = np.mean(local_mags) if local_mags else np.abs(audio_fft[bin_idx])
            
            # Calculate ratio and detect bit using adaptive thresholds
            if reference_magnitude > 0:
                ratio = current_magnitude / reference_magnitude
                
                # Use adaptive thresholds based on spectrum characteristics
                # Analyze local spectrum variation for threshold adjustment
                local_variation = np.std([np.abs(audio_fft[idx]) for idx in ref_bins]) if ref_bins else 0
                base_threshold = 1.0
                
                # Adaptive threshold based on embedding strength and local variation
                variation_factor = float(local_variation / reference_magnitude) if reference_magnitude > 0 else 0.0
                high_threshold = base_threshold + strength * 1.8  # Match stronger embedding
                low_threshold = base_threshold - strength * 1.2
                
                # Bit decision using robust thresholds that match embedding method
                # Embedding uses: bit '1' -> ref * (2.0 + strength * 2.0), bit '0' -> ref * (0.5 - strength * 0.4)
                expected_high = 2.0 + strength * 2.0
                expected_low = 0.5 - strength * 0.4
                
                # Use midpoint as decision threshold
                decision_threshold = (expected_high + expected_low) / 2.0
                
                if ratio > decision_threshold:
                    bit = '1'
                    # Confidence based on how close to expected '1' value
                    confidence = min(abs(ratio - expected_high) / expected_high, 1.0)
                    confidence = 1.0 - confidence  # Invert so closer = higher confidence
                else:
                    bit = '0'
                    # Confidence based on how close to expected '0' value
                    confidence = min(abs(ratio - expected_low) / max(expected_low, 0.1), 1.0)
                    confidence = 1.0 - confidence  # Invert so closer = higher confidence
                
                confidence = max(0.1, confidence)  # Minimum confidence
                
                extracted_binary += bit
                confidence_scores.append(max(0.1, confidence))  # Minimum confidence
            else:
                extracted_binary += '0'
                confidence_scores.append(0.1)
    
    # Convert binary to text with multiple attempts for robustness
    extracted_text = binary_to_text(extracted_binary)
    
    # If extraction failed, try with slight variations
    if not extracted_text and len(extracted_binary) >= 8:
        # Try shifting by 1 bit in case of alignment issues
        for shift in [1, -1, 2, -2]:
            if shift > 0:
                shifted_binary = '0' * shift + extracted_binary[:-shift]
            else:
                shifted_binary = extracted_binary[-shift:] + '0' * (-shift)
            
            candidate_text = binary_to_text(shifted_binary)
            if candidate_text and len(candidate_text) > len(extracted_text):
                extracted_text = candidate_text
                break
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
    
    return extracted_text, extracted_binary, avg_confidence

def plot_frequency_spectrum(audio_data, sample_rate, title, color="blue", highlight_bins=None):
    """Plot frequency spectrum with optional bin highlighting"""
    frequencies = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
    magnitude = np.abs(np.fft.rfft(audio_data))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frequencies, magnitude, color=color, linewidth=1, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    
    #Highlight watermark frequency bins
    if highlight_bins:
        for bin_idx in highlight_bins:
            if bin_idx < len(frequencies):
                freq = frequencies[bin_idx]
                ax.axvline(x=freq, color='red', alpha=0.6, linestyle='--', linewidth=1)
        ax.plot([], [], color='red', linestyle='--', label='Watermark Bins')
        ax.legend()
    
    st.pyplot(fig)

def plot_binary_comparison(original_binary, extracted_binary, title="Binary Comparison"):
    """Plot comparison between original and extracted binary"""
    if not original_binary or not extracted_binary:
        st.warning("No binary data to compare")
        return

    max_len = max(len(original_binary), len(extracted_binary))
    #Pad shorter string with zeros
    orig_padded = original_binary.ljust(max_len, '0')
    extr_padded = extracted_binary.ljust(max_len, '0')
    
    orig_bits = [int(b) for b in orig_padded[:min(100, max_len)]]  # Limit display
    extr_bits = [int(b) for b in extr_padded[:min(100, max_len)]]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    #Original binary
    ax1.step(range(len(orig_bits)), orig_bits, where='mid', linewidth=2, color='blue', label='Original')
    ax1.set_title('Original Binary')
    ax1.set_ylabel('Bit Value')
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3)
    
    #Extracted binary
    ax2.step(range(len(extr_bits)), extr_bits, where='mid', linewidth=2, color='red', label='Extracted')
    ax2.set_title('Extracted Binary')
    ax2.set_xlabel('Bit Index')
    ax2.set_ylabel('Bit Value')
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3)
    
    #Highlight differences
    for i, (orig, extr) in enumerate(zip(orig_bits, extr_bits)):
        if orig != extr:
            ax1.axvspan(i-0.5, i+0.5, color='yellow', alpha=0.3)
            ax2.axvspan(i-0.5, i+0.5, color='yellow', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def calculate_quality_metrics(original, watermarked):
    """Calculate audio quality metrics"""
    orig_float = original.astype(np.float64)
    wat_float = watermarked.astype(np.float64)
    
    # Ensure both arrays are the same length
    min_len = min(len(orig_float), len(wat_float))
    orig_float = orig_float[:min_len]
    wat_float = wat_float[:min_len]
    
    #Signal-to-Noise Ratio
    noise = orig_float - wat_float
    signal_power = np.mean(orig_float ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    #Peak Signal-to-Noise Ratio
    max_val = np.max(np.abs(orig_float))
    mse = np.mean((orig_float - wat_float) ** 2)
    
    if mse > 0:
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    #Normalized Cross-Correlation
    correlation = np.corrcoef(orig_float, wat_float)[0, 1]
    
    return snr, psnr, correlation

def create_download_link(audio_data, sample_rate, filename="watermarked_audio.wav"):
    """Create download link for audio file"""
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_data)
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def remove_watermark_fft(audio_data, sample_rate, params):
    audio_float = audio_data.astype(np.float64)
    audio_fft = np.fft.fft(audio_float)

    freq_start = params['freq_start']
    freq_width = params['freq_width']
    binary_length = params['binary_length']

    freq_resolution = sample_rate / len(audio_float)
    start_bin = int(freq_start / freq_resolution)
    end_bin = int((freq_start + freq_width) / freq_resolution)
    available_bins = list(range(start_bin, min(end_bin, len(audio_fft)//2)))
    step = max(3, len(available_bins) // binary_length)
    selected_bins = available_bins[::step][:binary_length]

    for bin_idx in selected_bins:
        ref_bins = [bin_idx + offset for offset in [-4, -3, 3, 4]
                    if 0 <= bin_idx + offset < len(audio_fft)//2 and bin_idx + offset not in selected_bins]
        if ref_bins:
            avg_mag = np.mean([np.abs(audio_fft[i]) for i in ref_bins])
            avg_phase = np.mean([np.angle(audio_fft[i]) for i in ref_bins])
            audio_fft[bin_idx] = avg_mag * np.exp(1j * avg_phase)
            neg_bin = len(audio_fft) - bin_idx
            if neg_bin != bin_idx and neg_bin < len(audio_fft):
                audio_fft[neg_bin] = np.conj(audio_fft[bin_idx])

    cleaned_audio = np.fft.ifft(audio_fft).real
    cleaned_audio = (cleaned_audio / np.max(np.abs(cleaned_audio)) * 32767).astype(np.int16)
    return cleaned_audio

def main():
    st.set_page_config(
        page_title="FFT Audio Watermarking Tool", 
        page_icon="🎵",
        layout="wide"
    )
    #Main header
    st.title("🎵 FFT Audio Watermarking Tool")
    st.markdown("**Text embedding and extraction using frequency domain manipulation**")
    st.markdown("---")

    st.sidebar.header("⚙️ Watermarking Parameters")
    
    freq_start = st.sidebar.slider("Start Frequency (Hz)", 1000, 8000, st.session_state.get("freq_start", 2000), 250)
    freq_width = st.sidebar.slider("Frequency Width (Hz)", 500, 4000, st.session_state.get("freq_width", 1500), 250)
    strength = st.sidebar.slider("Embedding Strength", 0.1, 1.0, st.session_state.get("strength", 0.3), 0.05)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""                
        **Parameter Guidelines:**
        - **Start Frequency**: 2000-4000 Hz (less perceptible range)
        - **Frequency Width**: Wider = more capacity
        - **Embedding Strength**: 0.3-0.7 for good balance
        
        **Tips:**
        - Dont use special characters and numbers
        - Use moderate strength for best results
        - Higher frequencies are less noticeable
                        
        **If Error:**
        - If it says "operands could not be broadcast together with shapes", convert first the audio to mono if stereo

        **For Extracting:**
        - Use the parameters that was used for embedding watermark
        - Adjust the embedding strength a little to get the accurate watermark 
    """)

    mode = st.selectbox("Select Operation Mode:", ["Embed Watermark", "Extract Watermark"])

    if mode == "Embed Watermark":
        st.header("📝 Embed Watermark")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload WAV audio file", 
                type=['wav'],
                help="Upload a WAV file to embed watermark"
            )
            
            watermark_text = st.text_input(
                "Enter watermark text", 
                max_chars=30,
                help="Enter text to embed (keep it short for better accuracy and DONT USE SPECIAL CHARACTERS)",
                placeholder="e.g., Thristian"
            )
            # Auto-Suggest Parameter Section
            if watermark_text:
                msg_len = len(watermark_text)

                if msg_len <= 6:
                    dynamic_freq_start = 2400
                    dynamic_freq_width = 1200
                    dynamic_strength = 0.3
                    category = "Short"
                elif msg_len == 7:
                    dynamic_freq_start = 3400
                    dynamic_freq_width = 1400
                    dynamic_strength = 0.5
                    category = "Short"
                elif msg_len == 8:
                    dynamic_freq_start = 2400
                    dynamic_freq_width = 1200
                    dynamic_strength = 0.3
                    category = "Short"
                elif 9 <= msg_len <= 16:
                    dynamic_freq_start = 4200
                    dynamic_freq_width = 2100
                    dynamic_strength = 0.55
                    category = "Medium"
                elif 17 <= msg_len <= 18:
                    dynamic_freq_start = 4600
                    dynamic_freq_width = 2600
                    dynamic_strength = 0.45
                    category = "Long"
                elif msg_len ==19:
                    dynamic_freq_start = 4500
                    dynamic_freq_width = 2400
                    dynamic_strength = 0.55
                    category = "Long"
                elif msg_len == 20:
                    dynamic_freq_start = 4200
                    dynamic_freq_width = 2200
                    dynamic_strength = 0.55
                    category = "Long"
                else:
                    dynamic_freq_start = 4200
                    dynamic_freq_width = 2500
                    dynamic_strength = 0.50
                    category = "Long"

                st.markdown("### 📐 Auto-Suggested Parameters")
                st.markdown(f"""
                | **Category** | **Message Length** | **Start Frequency** | **Width** | **Strength** |
                |--------------|--------------------|----------------------|-----------|--------------|
                | {category}   | {msg_len} chars     | {dynamic_freq_start} Hz   | {dynamic_freq_width} Hz  | {dynamic_strength} |
                """)
                
                if st.button("📌 Use Suggested Parameters"):
                    st.session_state.freq_start = dynamic_freq_start
                    st.session_state.freq_width = dynamic_freq_width
                    st.session_state.strength = dynamic_strength
                    st.success("✅ Suggested parameters applied! Please check the sidebar sliders.")
        with col2:
            if watermark_text:
                binary = text_to_binary(watermark_text)
                max_capacity = int(freq_width * 0.6 / 8)
                
                st.info(f"""
                **Text Analysis:**
                - Length: {len(watermark_text)} characters
                - Binary length: {len(binary)} bits
                - Estimated capacity: ~{max_capacity} characters
                - Status: {'✅ Good' if len(watermark_text) <= max_capacity else '⚠️ Too long'}
                """)
        
        if uploaded_file and watermark_text:
            try:
                audio_bytes = uploaded_file.read()
                sample_rate, audio_data = read(io.BytesIO(audio_bytes))
                
                st.success(f"Audio loaded: {len(audio_data):,} samples, {sample_rate:,} Hz, {len(audio_data)/sample_rate:.2f}s")
                
                with st.spinner("Embedding watermark..."):
                    watermarked_audio, params = embed_watermark_fft(
                        audio_data, sample_rate, watermark_text, 
                        freq_start, freq_width, strength
                    )
                
                st.success(f"✅ Watermark embedded successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quality Metrics")
                    snr, psnr, correlation = calculate_quality_metrics(audio_data, watermarked_audio)
                    
                    st.metric("Signal-to-Noise Ratio", f"{snr:.2f} dB")
                    st.metric("Peak SNR", f"{psnr:.2f} dB")  
                    st.metric("Correlation", f"{correlation:.4f}")
                
                with col2:
                    st.subheader("Embedding Parameters (Copy for Extraction)")
                    st.code(f"""
                    Watermark Text: {params['text']}
                    Start Frequency: {freq_start}
                    Frequency Width: {freq_width}
                    Embedding Strength: {strength}
                    Text Length: {len(params['text'])}
                    """, language="text")
                    
                    st.info("📝 Copy these exact parameters to extract the watermark from any audio file later.")
                
                # Store parameters in session state for extraction testing
                st.session_state['embedding_params'] = params
                st.session_state['original_audio'] = audio_data
                st.session_state['watermarked_audio'] = watermarked_audio
                st.session_state['sample_rate'] = sample_rate
                
                st.subheader("Download Files")
                col1, col2 = st.columns(2)
                
                with col1:
                    download_link = create_download_link(watermarked_audio, sample_rate, "watermarked_audio.wav")
                    st.markdown(download_link, unsafe_allow_html=True)
                
                with col2:
                    if st.button("Show Frequency Analysis"):
                        st.subheader("Frequency Domain Analysis")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            plot_frequency_spectrum(audio_data, sample_rate, "Original Audio Spectrum")
                        with col_b:
                            plot_frequency_spectrum(watermarked_audio, sample_rate, "Watermarked Audio Spectrum", 
                                                   highlight_bins=params['selected_bins'])
                
            except Exception as e:
                st.error(f"Error embedding watermark: {str(e)}")

    else:
        st.header("🔍 Extract Watermark")
        
        uploaded_file = st.file_uploader(
            "Upload watermarked WAV file", 
            type=['wav'],
            help="Upload a watermarked WAV file to extract the hidden text"
        )
        
        # Parameters for extraction
        st.subheader("Extraction Parameters")
        
        st.warning("⚠️ Enter the EXACT parameters that were used during embedding for accurate extraction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Manual Parameter Entry")
            extract_freq_start = st.number_input("Start Frequency (Hz)", value=2000, step=50, help="Must match embedding frequency")
            extract_freq_width = st.number_input("Frequency Width (Hz)", value=1500, step=50, help="Must match embedding width")
            extract_strength = st.number_input("Embedding Strength", value=0.3, step=0.05, help="Must match embedding strength")
            text_length = st.number_input("Expected Text Length", value=10, min_value=1, max_value=50, help="Length of embedded text")
            
            if st.checkbox("Use session parameters", help="Use parameters from current embedding session if available"):
                if 'embedding_params' in st.session_state:
                    params = st.session_state['embedding_params']
                    extract_freq_start = params['freq_start']
                    extract_freq_width = params['freq_width']
                    extract_strength = params['strength']
                    text_length = params['text_length']
                    st.info(f"Using session parameters: {params['text']} @ {extract_freq_start}Hz")
                else:
                    st.warning("No session parameters available")
        
        with col2:
            st.subheader("Current Parameters")
            st.write(f"**Start Frequency:** {extract_freq_start} Hz")
            st.write(f"**Frequency Width:** {extract_freq_width} Hz")
            st.write(f"**Embedding Strength:** {extract_strength}")
            st.write(f"**Expected Text Length:** {text_length} characters")
            st.write(f"**Binary Length:** {text_length * 8} bits")
            
            estimated_capacity = int(extract_freq_width * 0.6 / 8)
            if text_length <= estimated_capacity:
                st.success(f"Parameters look good for {text_length} character text")
            else:
                st.warning(f"Text length may be too long. Maximum recommended: {estimated_capacity} characters")
        
        if uploaded_file:
            try:
                audio_bytes = uploaded_file.read()
                sample_rate, audio_data = read(io.BytesIO(audio_bytes))
                
                st.success(f"Audio loaded: {len(audio_data):,} samples, {sample_rate:,} Hz")
                
                #Create extraction parameters from manual input
                binary_length = text_length * 8
                
                params = {
                    'binary_length': binary_length,
                    'freq_start': extract_freq_start,
                    'freq_width': extract_freq_width,
                    'strength': extract_strength,
                    'text_length': text_length
                }
                
                st.info(f"Using parameters: {extract_freq_start}Hz, {extract_freq_width}Hz width, {extract_strength} strength, {text_length} chars")

                with st.spinner("Extracting watermark..."):
                    extracted_text, extracted_binary, confidence = extract_watermark_fft(
                        audio_data, sample_rate, params
                    )
                
                #Display results
                st.subheader("Extraction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Extracted Text", f"'{extracted_text}'")
                    st.metric("Confidence Score", f"{confidence:.3f}")
                    st.metric("Binary Length", f"{len(extracted_binary)} bits")
                    
                    if confidence > 0.7:
                        st.success("High confidence extraction")
                    elif confidence > 0.4:
                        st.warning("Medium confidence extraction")
                    else:
                        st.error("Low confidence extraction")
                #Download Link for Cleaned audio
                st.subheader("Download Cleaned Audio (Watermark Removed)")
                cleaned_audio = remove_watermark_fft(audio_data, sample_rate, params)
                download_cleaned = create_download_link(cleaned_audio, sample_rate, "cleaned_audio.wav")
                st.markdown(download_cleaned, unsafe_allow_html=True)

                with col2:
                    st.subheader("Extraction Details")
                    st.write(f"**Frequency range:** {extract_freq_start}-{extract_freq_start + extract_freq_width} Hz")
                    st.write(f"**Text length:** {len(extracted_text)} characters")
                    st.write(f"**Binary:** {extracted_binary[:50]}{'...' if len(extracted_binary) > 50 else ''}")
                
                # Show frequency analysis
                if st.button("Show Extraction Analysis"):
                    st.subheader("Binary Analysis")
                    if 'embedding_params' in st.session_state:
                        original_binary = text_to_binary(st.session_state['embedding_params']['text'])
                        plot_binary_comparison(original_binary, extracted_binary)

                        match_count = sum(o == e for o, e in zip(original_binary, extracted_binary))
                        total = min(len(original_binary), len(extracted_binary))
                        bit_accuracy = (match_count / total) * 100 if total > 0 else 0
                        st.metric("Bit Accuracy", f"{bit_accuracy:.2f}%")
                    
                    else:
                        st.info("Upload original embedding parameters to compare binary data")
                
            except Exception as e:
                st.error(f"Error extracting watermark: {str(e)}")

if __name__ == "__main__":
    main()
