import os
import gradio as gr
import json
import time
import tempfile
from typing import List, Dict, Any
from pathlib import Path
import traceback

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import your agent (adjust path based on your structure)
import sys
sys.path.append('./first-draft-HuggingFaceHub/innovation-catalyst-ai-agent/src')

from innovation_catalyst.agents.innovation_agent import get_innovation_agent
from innovation_catalyst.utils.logging import get_logger

logger = get_logger(__name__)

class InnovationCatalystApp:
    """Production-ready Gradio app for Innovation Catalyst Agent."""
    
    def __init__(self):
        # Validate environment
        if not os.getenv("GITHUB_API_KEY"):
            raise ValueError("❌ GITHUB_API_KEY not found. Please set it in environment variables.")
        
        try:
            self.agent = get_innovation_agent()
            logger.info("✅ Agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
        
        self.max_file_size = 5 * 1024 * 1024  # 5MB (reduced for stability)
        self.supported_formats = ['.pdf', '.txt', '.docx', '.md']
        self.max_files = 3  # Limit files to prevent token overflow
        
    def process_uploaded_files(
        self, 
        files: List[str], 
        focus_theme: str = "innovation",
        max_connections: int = 10  # Reduced default
    ) -> tuple:
        """Process uploaded files with robust error handling."""
        
        if not files:
            return (
                "❌ No files uploaded", 
                "Please upload at least one document (PDF, TXT, DOCX, or MD).", 
                0.0
            )
        
        if len(files) > self.max_files:
            return (
                f"❌ Too many files", 
                f"Please upload maximum {self.max_files} files to stay within processing limits.", 
                0.0
            )
        
        try:
            start_time = time.time()
            
            # Process files with size validation
            file_data = []
            for file_path in files:
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > self.max_file_size:
                        continue
                    
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    # Convert to string for text processing
                    try:
                        if Path(file_path).suffix.lower() == '.pdf':
                            content_str = content  # Keep as bytes for PDF
                        else:
                            content_str = content.decode('utf-8', errors='ignore')
                    except Exception:
                        content_str = str(content, errors='ignore')
                    
                    file_data.append({
                        'name': Path(file_path).name,
                        'content': content_str,
                        'size': file_size,
                        'format': Path(file_path).suffix.lower()
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue
            
            if not file_data:
                return (
                    "❌ No valid files", 
                    "All uploaded files were invalid, too large, or corrupted.", 
                    0.0
                )
            
            # Process with the agent
            logger.info(f"Processing {len(file_data)} files with theme '{focus_theme}'")
            
            result = self.agent.process_documents(
                file_data, 
                focus_theme=focus_theme, 
                max_connections=min(max_connections, 15)  # Cap connections
            )
            
            processing_time = time.time() - start_time
            
            # Format results
            if result.get("success", False):
                success_msg = f"✅ Processed {len(file_data)} documents in {processing_time:.1f}s"
                detailed_results = self._format_results(result)
            else:
                success_msg = f"⚠️ Processing issues: {result.get('error_message', 'Unknown error')}"
                detailed_results = self._format_error_results(result)
            
            return success_msg, detailed_results, processing_time
            
        except Exception as e:
            error_msg = f"❌ Processing failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return error_msg, f"Error details: {str(e)}", 0.0
    
    def _format_results(self, result: Dict[str, Any]) -> str:
        """Format successful results for display."""
        output = ["# 🚀 Innovation Catalyst Results\n"]
        
        # Processing Summary
        output.append("## 📊 Processing Summary")
        output.append(f"- **Processing Time**: {result.get('processing_time', 0):.2f} seconds")
        output.append(f"- **Innovation Score**: {result.get('innovation_score', 0):.2f}/1.0")
        output.append(f"- **Provider**: GitHub Models (Free!) 🎓\n")
        
        # Key Insights
        if 'summary' in result:
            output.append("## 🎯 Executive Summary")
            output.append(result['summary'])
            output.append("")
        
        # Innovation Opportunities
        if 'innovation_insights' in result and result['innovation_insights']:
            output.append("## 💡 Innovation Opportunities")
            for i, insight in enumerate(result['innovation_insights'][:5], 1):
                output.append(f"{i}. {insight}")
            output.append("")
        
        # Actionable Steps
        if 'actionable_steps' in result and result['actionable_steps']:
            output.append("## 🎯 Next Steps")
            for i, step in enumerate(result['actionable_steps'][:6], 1):
                output.append(f"{i}. {step}")
            output.append("")
        
        # Raw output (truncated)
        if 'raw_output' in result:
            output.append("## 📄 Detailed Analysis")
            output.append("```
            output.append(result['raw_output'][:1000])
            if len(result['raw_output']) > 1000:
                output.append("\n... (truncated for display)")
            output.append("```")
        
        return "\n".join(output)
    
    def _format_error_results(self, result: Dict[str, Any]) -> str:
        """Format error results."""
        output = ["# ⚠️ Processing Issues\n"]
        
        output.append("## Error Details")
        output.append(f"**Error**: {result.get('error_message', 'Unknown error')}")
        
        output.append("\n## Troubleshooting")
        output.append("- Try smaller files (< 5MB each)")
        output.append("- Upload fewer documents (max 3)")
        output.append("- Ensure files contain readable text")
        output.append("- Check file formats (PDF, TXT, DOCX, MD)")
        
        return "\n".join(output)

def create_app():
    """Create the Gradio interface."""
    try:
        app = InnovationCatalystApp()
        
        # Custom CSS
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto;
        }
        .output-markdown {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 8px;
        }
        """
        
        interface = gr.Interface(
            fn=app.process_uploaded_files,
            inputs=[
                gr.Files(
                    label="📁 Upload Documents (Max 3 files, 5MB each)",
                    file_types=[".pdf", ".txt", ".docx", ".md"],
                    file_count="multiple"
                ),
                gr.Textbox(
                    value="innovation",
                    label="🎯 Focus Theme",
                    placeholder="e.g., innovation, technology, healthcare, business",
                    info="What aspect should the analysis focus on?"
                ),
                gr.Slider(
                    minimum=5,
                    maximum=15,
                    value=10,
                    step=1,
                    label="🔗 Max Connections",
                    info="Maximum semantic connections to discover"
                )
            ],
            outputs=[
                gr.Textbox(label="📋 Status", interactive=False),
                gr.Markdown(label="📊 Results", elem_classes=["output-markdown"]),
                gr.Number(label="⏱️ Processing Time (seconds)", interactive=False)
            ],
            title="🚀 Innovation Catalyst Agent",
            description="""
            **Discover breakthrough innovation opportunities from your documents!**
            
            🎓 **Powered by GitHub Models (Free for Students!)**
            
            Upload documents and let AI discover novel connections between ideas to generate actionable innovation insights.
            
            **Supported formats**: PDF, TXT, DOCX, Markdown
            """,
            article="""
            ### 🔧 How It Works
            1. **Document Analysis**: Extracts and processes text from your uploads
            2. **Semantic Understanding**: Identifies entities, topics, and concepts  
            3. **Connection Discovery**: Finds novel relationships between ideas
            4. **Innovation Synthesis**: Generates actionable recommendations
            
            ### 💡 Tips for Best Results
            - Upload documents from different domains for cross-domain insights
            - Keep files under 5MB for optimal processing
            - Try different focus themes to explore various angles
            - Include both technical and business content
            
            ### 🚀 Built with SmolAgent Framework
            Professional AI agent architecture for reliable, scalable deployment.
            """,
            examples=[
                [None, "technology innovation", 10],
                [None, "sustainable business", 8],
                [None, "healthcare transformation", 12]
            ],
            css=css,
            theme=gr.themes.Soft(),
            allow_flagging="never"
        )
        
        return interface
        
    except Exception as e:
        # Fallback interface for setup errors
        def error_interface():
            return f"❌ Setup Error: {str(e)}", "Please check your configuration and try again.", 0.0
        
        return gr.Interface(
            fn=error_interface,
            inputs=[],
            outputs=[
                gr.Textbox(label="Status"),
                gr.Textbox(label="Error Details"),
                gr.Number(label="Time")
            ],
            title="🚀 Innovation Catalyst Agent - Configuration Error"
        )

# Launch the app
if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )