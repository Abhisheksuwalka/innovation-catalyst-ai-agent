# src/innovation_catalyst/tools/innovation_synthesis.py
"""
Innovation synthesis tool that generates comprehensive innovation insights.
Implements advanced synthesis generation with LLM integration and structured output.
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

# SmolAgent integration
from smolagents import tool

from ..models.connections import SemanticConnection, InnovationPotential
from ..utils.config import get_config
from ..utils.logging import get_logger, log_function_performance

logger = get_logger(__name__)
config = get_config()

@dataclass
class SynthesisResult:
    """Result of innovation synthesis operation."""
    synthesis_text: str
    innovation_score: float
    actionable_steps: List[str]
    potential_applications: List[str]
    risk_factors: List[str]
    technical_feasibility: float
    market_potential: str
    resource_requirements: Dict[str, Any]
    competitive_advantage: str
    implementation_timeline: str
    processing_time: float
    confidence_score: float = 0.0
    error_message: Optional[str] = None

class InnovationSynthesizer:
    """
    Advanced innovation synthesis engine.
    
    Features:
        - Connection pattern analysis
        - Innovation opportunity identification
        - Structured synthesis generation
        - Feasibility assessment
        - Risk evaluation
        - Implementation guidance
    """
    
    def __init__(self):
        self.synthesis_templates = self._load_synthesis_templates()
        self.domain_expertise = self._load_domain_expertise()
        
        logger.info("InnovationSynthesizer initialized")
    
    def _load_synthesis_templates(self) -> Dict[str, str]:
        """Load synthesis templates for different innovation types."""
        return {
            "cross_domain": """
            **Cross-Domain Innovation Synthesis**
            
            This synthesis combines insights from {domain_1} and {domain_2}, creating novel opportunities for innovation.
            
            **Key Innovation Insights:**
            {key_insights}
            
            **Novel Connections Identified:**
            {novel_connections}
            
            **Innovation Potential:**
            The combination of {shared_concepts} across different domains presents significant innovation opportunities.
            {innovation_rationale}
            """,
            
            "technology_focused": """
            **Technology Innovation Synthesis**
            
            This synthesis focuses on technological innovations emerging from the analysis of {technology_areas}.
            
            **Technical Innovations:**
            {technical_innovations}
            
            **Implementation Approaches:**
            {implementation_approaches}
            
            **Technology Readiness:**
            {technology_readiness}
            """,
            
            "business_focused": """
            **Business Innovation Synthesis**
            
            This synthesis identifies business model innovations and market opportunities.
            
            **Business Model Innovations:**
            {business_innovations}
            
            **Market Opportunities:**
            {market_opportunities}
            
            **Value Proposition:**
            {value_proposition}
            """,
            
            "general": """
            **Innovation Synthesis**
            
            Based on the analysis of connections between different concepts, several innovation opportunities emerge.
            
            **Key Findings:**
            {key_findings}
            
            **Innovation Opportunities:**
            {innovation_opportunities}
            
            **Strategic Implications:**
            {strategic_implications}
            """
        }
    
    def _load_domain_expertise(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific expertise for synthesis."""
        return {
            "technology": {
                "key_factors": ["scalability", "performance", "security", "usability"],
                "risk_factors": ["technical complexity", "integration challenges", "obsolescence risk"],
                "success_metrics": ["adoption rate", "performance improvement", "cost reduction"]
            },
            "business": {
                "key_factors": ["market size", "competitive advantage", "revenue model", "customer value"],
                "risk_factors": ["market competition", "regulatory changes", "customer adoption"],
                "success_metrics": ["revenue growth", "market share", "customer satisfaction"]
            },
            "healthcare": {
                "key_factors": ["patient outcomes", "safety", "regulatory compliance", "cost effectiveness"],
                "risk_factors": ["regulatory approval", "safety concerns", "adoption barriers"],
                "success_metrics": ["patient outcomes", "cost savings", "efficiency gains"]
            },
            "finance": {
                "key_factors": ["risk management", "regulatory compliance", "customer trust", "efficiency"],
                "risk_factors": ["regulatory changes", "security breaches", "market volatility"],
                "success_metrics": ["cost reduction", "risk mitigation", "customer acquisition"]
            }
        }
    
    @log_function_performance("innovation_synthesis")
    def generate_synthesis(
        self,
        connections: List[Dict[str, Any]],
        focus_theme: str = "innovation",
        synthesis_type: str = "general"
    ) -> SynthesisResult:
        """
        Generate comprehensive innovation synthesis.
        
        Args:
            connections (List[Dict[str, Any]]): Top connections to synthesize
            focus_theme (str): Theme to focus synthesis on
            synthesis_type (str): Type of synthesis to generate
            
        Returns:
            SynthesisResult: Complete synthesis results
        """
        start_time = time.time()
        
        try:
            if not connections:
                return SynthesisResult(
                    synthesis_text="No connections provided for synthesis.",
                    innovation_score=0.0,
                    actionable_steps=[],
                    potential_applications=[],
                    risk_factors=["No data available for analysis"],
                    technical_feasibility=0.0,
                    market_potential="Unknown",
                    resource_requirements={},
                    competitive_advantage="None identified",
                    implementation_timeline="Cannot be determined",
                    processing_time=time.time() - start_time,
                    error_message="No connections provided"
                )
            
            logger.info(f"Generating synthesis for {len(connections)} connections")
            
            # Analyze connection patterns
            patterns = self._analyze_connection_patterns(connections)
            
            # Generate synthesis text
            synthesis_text = self._generate_synthesis_text(
                connections, patterns, focus_theme, synthesis_type
            )
            
            # Calculate innovation score
            innovation_score = self._calculate_innovation_score(connections, patterns)
            
            # Generate actionable steps
            actionable_steps = self._generate_actionable_steps(connections, patterns)
            
            # Identify potential applications
            potential_applications = self._identify_potential_applications(connections, patterns)
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(connections, patterns)
            
            # Evaluate technical feasibility
            technical_feasibility = self._evaluate_technical_feasibility(connections, patterns)
            
            # Assess market potential
            market_potential = self._assess_market_potential(connections, patterns)
            
            # Determine resource requirements
            resource_requirements = self._determine_resource_requirements(connections, patterns)
            
            # Identify competitive advantage
            competitive_advantage = self._identify_competitive_advantage(connections, patterns)
            
            # Create implementation timeline
            implementation_timeline = self._create_implementation_timeline(connections, patterns)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(connections, patterns)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Synthesis generated successfully in {processing_time:.2f}s")
            
            return SynthesisResult(
                synthesis_text=synthesis_text,
                innovation_score=innovation_score,
                actionable_steps=actionable_steps,
                potential_applications=potential_applications,
                risk_factors=risk_factors,
                technical_feasibility=technical_feasibility,
                market_potential=market_potential,
                resource_requirements=resource_requirements,
                competitive_advantage=competitive_advantage,
                implementation_timeline=implementation_timeline,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Synthesis generation failed: {str(e)}"
            logger.error(error_msg)
            
            return SynthesisResult(
                synthesis_text="Failed to generate synthesis.",
                innovation_score=0.0,
                actionable_steps=[],
                potential_applications=[],
                risk_factors=["Synthesis generation failed"],
                technical_feasibility=0.0,
                market_potential="Unknown",
                resource_requirements={},
                competitive_advantage="None identified",
                implementation_timeline="Cannot be determined",
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def _analyze_connection_patterns(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in the connections."""
        patterns = {
            "dominant_themes": [],
            "cross_domain_connections": [],
            "high_innovation_connections": [],
            "shared_entities": [],
            "shared_topics": [],
            "connection_types": [],
            "innovation_levels": []
        }
        
        # Collect all shared elements
        all_entities = []
        all_topics = []
        all_keywords = []
        
        for connection in connections:
            all_entities.extend(connection.get("shared_entities", []))
            all_topics.extend(connection.get("shared_topics", []))
            all_keywords.extend(connection.get("shared_keywords", []))
            
            patterns["connection_types"].append(connection.get("connection_type", "unknown"))
            patterns["innovation_levels"].append(connection.get("innovation_level", "low"))
            
            # Identify high innovation connections
            if connection.get("innovation_potential", 0) > 0.7:
                patterns["high_innovation_connections"].append(connection)
        
        # Find dominant themes
        from collections import Counter
        entity_counts = Counter(all_entities)
        topic_counts = Counter(all_topics)
        keyword_counts = Counter(all_keywords)
        
        patterns["dominant_themes"] = [
            item for item, count in topic_counts.most_common(5)
        ]
        patterns["shared_entities"] = [
            item for item, count in entity_counts.most_common(10)
        ]
        patterns["shared_topics"] = [
            item for item, count in topic_counts.most_common(10)
        ]
        
        return patterns
    
    def _generate_synthesis_text(
        self,
        connections: List[Dict[str, Any]],
        patterns: Dict[str, Any],
        focus_theme: str,
        synthesis_type: str
    ) -> str:
        """Generate the main synthesis text."""
        template = self.synthesis_templates.get(synthesis_type, self.synthesis_templates["general"])
        
        # Prepare template variables
        template_vars = {
            "key_insights": self._generate_key_insights(connections, patterns),
            "novel_connections": self._describe_novel_connections(connections, patterns),
            "innovation_rationale": self._generate_innovation_rationale(connections, patterns),
            "shared_concepts": ", ".join(patterns["shared_topics"][:3]),
            "domain_1": patterns["shared_topics"][0] if patterns["shared_topics"] else "Technology",
            "domain_2": patterns["shared_topics"][1] if len(patterns["shared_topics"]) > 1 else "Business",
            "key_findings": self._generate_key_findings(connections, patterns),
            "innovation_opportunities": self._describe_innovation_opportunities(connections, patterns),
            "strategic_implications": self._generate_strategic_implications(connections, patterns)
        }
        
        # Format template
        try:
            synthesis_text = template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            synthesis_text = self._generate_fallback_synthesis(connections, patterns)
        
        return synthesis_text
    
    def _generate_key_insights(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Generate key insights from connections."""
        insights = []
        
        if patterns["high_innovation_connections"]:
            insights.append(f"• {len(patterns['high_innovation_connections'])} high-potential innovation opportunities identified")
        
        if patterns["dominant_themes"]:
            insights.append(f"• Key themes emerging: {', '.join(patterns['dominant_themes'][:3])}")
        
        if patterns["shared_entities"]:
            insights.append(f"• Common entities across connections: {', '.join(patterns['shared_entities'][:3])}")
        
        # Calculate average innovation potential
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        insights.append(f"• Average innovation potential: {avg_innovation:.2f}")
        
        return "\n".join(insights) if insights else "No specific insights identified."
    
    def _describe_novel_connections(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Describe the most novel connections."""
        novel_descriptions = []
        
        # Sort connections by novelty score
        sorted_connections = sorted(
            connections, 
            key=lambda x: x.get("novelty_score", 0), 
            reverse=True
        )
        
        for i, connection in enumerate(sorted_connections[:3]):
            description = f"{i+1}. {connection.get('explanation', 'Connection identified')}"
            if connection.get("novelty_score", 0) > 0.7:
                description += " (High novelty)"
            novel_descriptions.append(description)
        
        return "\n".join(novel_descriptions) if novel_descriptions else "No novel connections identified."
    
    def _generate_innovation_rationale(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Generate rationale for innovation potential."""
        rationale_points = []
        
        # Cross-domain analysis
        unique_topics = set(patterns["shared_topics"])
        if len(unique_topics) > 2:
            rationale_points.append("Cross-domain connections enable novel solution approaches")
        
        # High innovation connections
        high_innovation_count = len(patterns["high_innovation_connections"])
        if high_innovation_count > 0:
            rationale_points.append(f"{high_innovation_count} connections show breakthrough potential")
        
        # Shared entities creating new possibilities
        if len(patterns["shared_entities"]) > 3:
            rationale_points.append("Multiple shared entities create rich innovation possibilities")
        
        return ". ".join(rationale_points) + "." if rationale_points else "Innovation potential exists through identified connections."
    
    def _generate_key_findings(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Generate key findings summary."""
        findings = []
        
        # Connection analysis
        findings.append(f"Analyzed {len(connections)} semantic connections")
        
        # Innovation distribution
        innovation_levels = patterns["innovation_levels"]
        high_innovation = innovation_levels.count("high") + innovation_levels.count("breakthrough")
        if high_innovation > 0:
            findings.append(f"{high_innovation} connections show high innovation potential")
        
        # Theme analysis
        if patterns["dominant_themes"]:
            findings.append(f"Dominant themes: {', '.join(patterns['dominant_themes'][:3])}")
        
        return ". ".join(findings) + "."
    
    def _describe_innovation_opportunities(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Describe specific innovation opportunities."""
        opportunities = []
        
        # High-potential connections
        for connection in patterns["high_innovation_connections"][:3]:
            opportunity = f"• {connection.get('explanation', 'Innovation opportunity identified')}"
            opportunities.append(opportunity)
        
        # Cross-domain opportunities
        if len(set(patterns["shared_topics"])) > 2:
            opportunities.append("• Cross-domain integration opportunities")
        
        # Entity-based opportunities
        if patterns["shared_entities"]:
            opportunities.append(f"• Leverage shared entities: {', '.join(patterns['shared_entities'][:2])}")
        
        return "\n".join(opportunities) if opportunities else "Innovation opportunities require further analysis."
    
    def _generate_strategic_implications(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Generate strategic implications."""
        implications = []
        
        # Innovation readiness
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        if avg_innovation > 0.6:
            implications.append("High innovation readiness suggests immediate action potential")
        
        # Cross-domain implications
        if len(set(patterns["shared_topics"])) > 2:
            implications.append("Cross-domain connections require interdisciplinary collaboration")
        
        # Resource implications
        if patterns["high_innovation_connections"]:
            implications.append("High-potential connections warrant priority resource allocation")
        
        return ". ".join(implications) + "." if implications else "Strategic implications require further analysis."
    
    def _generate_fallback_synthesis(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Generate fallback synthesis when template fails."""
        return f"""
        **Innovation Synthesis**
        
        Analysis of {len(connections)} connections reveals several innovation opportunities.
        
        **Key Themes:** {', '.join(patterns['dominant_themes'][:3]) if patterns['dominant_themes'] else 'Various themes identified'}
        
        **High-Potential Connections:** {len(patterns['high_innovation_connections'])} connections show significant innovation potential.
        
        **Shared Elements:** Common entities and topics create opportunities for novel combinations and applications.
        
        **Innovation Potential:** The identified connections suggest opportunities for breakthrough innovations through creative combination of existing concepts.
        """
    
    def _calculate_innovation_score(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> float:
        """Calculate overall innovation score."""
        if not connections:
            return 0.0
        
        # Base score from average innovation potential
        base_score = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        
        # Bonus for high innovation connections
        high_innovation_bonus = len(patterns["high_innovation_connections"]) * 0.1
        
        # Bonus for cross-domain connections
        cross_domain_bonus = min(len(set(patterns["shared_topics"])) * 0.05, 0.2)
        
        # Bonus for novelty
        avg_novelty = sum(conn.get("novelty_score", 0) for conn in connections) / len(connections)
        novelty_bonus = avg_novelty * 0.2
        
        total_score = base_score + high_innovation_bonus + cross_domain_bonus + novelty_bonus
        
        return min(1.0, total_score)
    
    def _generate_actionable_steps(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> List[str]:
        """Generate actionable implementation steps."""
        steps = []
        
        # Research and validation steps
        if patterns["high_innovation_connections"]:
            steps.append("Conduct detailed feasibility analysis for high-potential connections")
        
        steps.append("Validate key assumptions through market research and expert consultation")
        
        # Development steps
        if patterns["shared_entities"]:
            steps.append(f"Develop proof-of-concept focusing on {patterns['shared_entities'][0]}")
        
        # Cross-domain steps
        if len(set(patterns["shared_topics"])) > 2:
            steps.append("Establish cross-functional teams to explore interdisciplinary opportunities")
        
        # Implementation steps
        steps.append("Create detailed implementation roadmap with milestones and success metrics")
        steps.append("Identify and engage key stakeholders and potential partners")
        steps.append("Develop risk mitigation strategies for identified challenges")
        
        return steps
    
    def _identify_potential_applications(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> List[str]:
        """Identify potential applications for the innovations."""
        applications = []
        
        # Domain-specific applications
        domains = set(patterns["shared_topics"])
        
        if "technology" in [topic.lower() for topic in domains]:
            applications.append("Technology platform development")
            applications.append("Software solution creation")
        
        if "healthcare" in [topic.lower() for topic in domains]:
            applications.append("Healthcare service innovation")
            applications.append("Medical device development")
        
        if "business" in [topic.lower() for topic in domains]:
            applications.append("Business model innovation")
            applications.append("Process optimization solutions")
        
        if "finance" in [topic.lower() for topic in domains]:
            applications.append("Financial service innovation")
            applications.append("Risk management solutions")
        
        # Generic applications
        applications.extend([
            "Research and development initiatives",
            "Strategic partnership opportunities",
            "New product/service development",
            "Process improvement implementations"
        ])
        
        return applications[:8]  # Limit to top 8 applications
    
    def _assess_risk_factors(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> List[str]:
        """Assess potential risk factors."""
        risks = []
        
        # Innovation-specific risks
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        if avg_innovation > 0.8:
            risks.append("High innovation potential may face adoption resistance")
        
        # Cross-domain risks
        if len(set(patterns["shared_topics"])) > 3:
            risks.append("Cross-domain complexity may increase implementation challenges")
        
        # General risks
        risks.extend([
            "Market acceptance uncertainty",
            "Technical implementation challenges",
            "Resource allocation requirements",
            "Competitive response risks",
            "Regulatory compliance considerations"
        ])
        
        # Novelty-based risks
        avg_novelty = sum(conn.get("novelty_score", 0) for conn in connections) / len(connections)
        if avg_novelty > 0.7:
            risks.append("High novelty may require significant market education")
        
        return risks[:6]  # Limit to top 6 risks
    
    def _evaluate_technical_feasibility(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> float:
        """Evaluate technical feasibility score."""
        # Base feasibility from connection quality
        avg_quality = sum(conn.get("quality_score", 0.5) for conn in connections) / len(connections)
        
        # Adjust for complexity
        complexity_penalty = min(len(set(patterns["shared_topics"])) * 0.05, 0.2)
        
        # Adjust for innovation level
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        innovation_penalty = avg_innovation * 0.1  # Higher innovation = lower immediate feasibility
        
        feasibility = avg_quality - complexity_penalty - innovation_penalty
        
        return max(0.1, min(1.0, feasibility))
    
    def _assess_market_potential(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Assess market potential."""
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        
        if avg_innovation > 0.8:
            return "High - Breakthrough innovation potential with significant market impact"
        elif avg_innovation > 0.6:
            return "Medium-High - Strong innovation potential with good market opportunities"
        elif avg_innovation > 0.4:
            return "Medium - Moderate innovation potential with niche market opportunities"
        else:
            return "Low-Medium - Limited innovation potential with uncertain market impact"
    
    def _determine_resource_requirements(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Determine resource requirements."""
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        complexity = len(set(patterns["shared_topics"]))
        
        # Scale requirements based on innovation level and complexity
        base_requirements = {
            "human_resources": "Medium",
            "financial_investment": "Medium",
            "time_investment": "6-12 months",
            "technical_expertise": "Medium",
            "partnerships": "Recommended"
        }
        
        if avg_innovation > 0.7:
            base_requirements["human_resources"] = "High"
            base_requirements["financial_investment"] = "High"
            base_requirements["time_investment"] = "12-24 months"
        
        if complexity > 3:
            base_requirements["technical_expertise"] = "High"
            base_requirements["partnerships"] = "Essential"
        
        return base_requirements
    
    def _identify_competitive_advantage(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Identify potential competitive advantage."""
        advantages = []
        
        # Innovation-based advantages
        high_innovation_count = len(patterns["high_innovation_connections"])
        if high_innovation_count > 0:
            advantages.append("First-mover advantage in innovative solution space")
        
        # Cross-domain advantages
        if len(set(patterns["shared_topics"])) > 2:
            advantages.append("Unique cross-domain integration capabilities")
        
        # Novelty advantages
        avg_novelty = sum(conn.get("novelty_score", 0) for conn in connections) / len(connections)
        if avg_novelty > 0.7:
            advantages.append("Novel approach differentiation")
        
        # Entity-based advantages
        if len(patterns["shared_entities"]) > 3:
            advantages.append("Comprehensive entity relationship understanding")
        
        return ". ".join(advantages) if advantages else "Competitive advantage requires further analysis and development."
    
    def _create_implementation_timeline(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """Create implementation timeline."""
        avg_innovation = sum(conn.get("innovation_potential", 0) for conn in connections) / len(connections)
        complexity = len(set(patterns["shared_topics"]))
        
        if avg_innovation > 0.8 or complexity > 4:
            return "18-36 months for full implementation with 6-month proof-of-concept phase"
        elif avg_innovation > 0.6 or complexity > 2:
            return "12-24 months for full implementation with 3-month proof-of-concept phase"
        else:
            return "6-18 months for full implementation with 2-month proof-of-concept phase"
    
    def _calculate_confidence_score(self, connections: List[Dict[str, Any]], patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for the synthesis."""
        # Base confidence from connection quality
        avg_confidence = sum(conn.get("confidence_score", 0.5) for conn in connections) / len(connections)
        
        # Adjust for number of connections
        connection_bonus = min(len(connections) * 0.05, 0.2)
        
        # Adjust for high innovation connections
        high_innovation_bonus = len(patterns["high_innovation_connections"]) * 0.1
        
        confidence = avg_confidence + connection_bonus + high_innovation_bonus
        
        return min(1.0, confidence)

# Global innovation synthesizer instance
innovation_synthesizer = InnovationSynthesizer()

@tool
def generate_innovation_synthesis(connections: List[Dict[str, Any]], focus_theme: str = "innovation") -> Dict[str, Any]:
    """
    Generate comprehensive innovation synthesis from discovered connections.
    
    Args:
        connections (List[Dict]): Top connections to synthesize
        focus_theme (str): Theme to focus synthesis on
        
    Returns:
        Dict[str, Any]: Innovation synthesis with structure:
        {
            "synthesis_text": str,              # Main synthesis description
            "innovation_score": float,          # 0.0 to 1.0
            "actionable_steps": List[str],      # Implementation steps
            "potential_applications": List[str], # Use cases
            "risk_factors": List[str],          # Potential risks
            "technical_feasibility": float,     # 0.0 to 1.0
            "market_potential": str,            # Market assessment
            "resource_requirements": Dict,      # Resources needed
            "competitive_advantage": str,       # Unique value prop
            "implementation_timeline": str      # Estimated timeline
        }
        
    Synthesis Process:
        1. Analyze connection patterns and themes
        2. Identify innovation opportunities
        3. Generate structured synthesis using LLM
        4. Extract actionable components
        5. Assess feasibility and risks
        6. Provide implementation guidance
        
    LLM Prompting Strategy:
        - System prompt defines innovation expert persona
        - Include connection details and context
        - Request structured output with specific components
        - Use examples for consistency
    """
    result = innovation_synthesizer.generate_synthesis(connections, focus_theme)
    
    if result.error_message:
        logger.error(f"Innovation synthesis failed: {result.error_message}")
    
    return {
        "synthesis_text": result.synthesis_text,
        "innovation_score": result.innovation_score,
        "actionable_steps": result.actionable_steps,
        "potential_applications": result.potential_applications,
        "risk_factors": result.risk_factors,
        "technical_feasibility": result.technical_feasibility,
        "market_potential": result.market_potential,
        "resource_requirements": result.resource_requirements,
        "competitive_advantage": result.competitive_advantage,
        "implementation_timeline": result.implementation_timeline,
        "processing_time": result.processing_time,
        "confidence_score": result.confidence_score
    }

def get_synthesis_info() -> Dict[str, Any]:
    """Get information about synthesis capabilities."""
    return {
        "synthesis_types": ["general", "cross_domain", "technology_focused", "business_focused"],
        "focus_themes": ["innovation", "technology", "business", "research"],
        "output_components": [
            "synthesis_text", "innovation_score", "actionable_steps",
            "potential_applications", "risk_factors", "technical_feasibility",
            "market_potential", "resource_requirements", "competitive_advantage",
            "implementation_timeline"
        ]
    }
