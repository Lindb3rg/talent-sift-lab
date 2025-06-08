import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResumeDataVisualizer:
    def __init__(self, jsonl_file_path):
        self.data = self.load_jsonl_data(jsonl_file_path)
        self.df = self.create_dataframes()
    
    def load_jsonl_data(self, file_path):
        """Load JSONL data into a list of dictionaries"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data
    
    def create_dataframes(self):
        """Create structured DataFrames from the nested JSON data"""
        dataframes = {}
        
        # Main candidate info
        candidates = []
        for i, record in enumerate(self.data):
            personal_info = record.get('personal_info', {})
            location = personal_info.get('location', {})
            
            candidate = {
                'candidate_id': i,
                'name': personal_info.get('name', 'Unknown'),
                'email': personal_info.get('email', 'Unknown'),
                'phone': personal_info.get('phone', 'Unknown'),
                'city': location.get('city', 'Unknown'),
                'country': location.get('country', 'Unknown'),
                'remote_preference': location.get('remote_preference', 'Unknown'),
                'summary': personal_info.get('summary', 'Unknown'),
                'linkedin': personal_info.get('linkedin', 'Unknown'),
                'github': personal_info.get('github', 'Unknown')
            }
            candidates.append(candidate)
        dataframes['candidates'] = pd.DataFrame(candidates)
        
        # Experience data
        experiences = []
        for i, record in enumerate(self.data):
            experience_list = record.get('experience', [])
            for exp in experience_list:
                company_info = exp.get('company_info', {})
                dates = exp.get('dates', {})
                tech_env = exp.get('technical_environment', {})
                
                experience = {
                    'candidate_id': i,
                    'company': exp.get('company', 'Unknown'),
                    'title': exp.get('title', 'Unknown'),
                    'level': exp.get('level', 'Unknown'),
                    'employment_type': exp.get('employment_type', 'Unknown'),
                    'start_date': dates.get('start', 'Unknown'),
                    'end_date': dates.get('end', 'Unknown'),
                    'duration': dates.get('duration', 'Unknown'),
                    'industry': company_info.get('industry', 'Unknown'),
                    'company_size': company_info.get('size', 'Unknown'),
                    'technologies': ', '.join(tech_env.get('technologies', [])),
                    'tools': ', '.join(tech_env.get('tools', []))
                }
                experiences.append(experience)
        dataframes['experiences'] = pd.DataFrame(experiences)
        
        # Education data
        educations = []
        for i, record in enumerate(self.data):
            education_list = record.get('education', [])
            for edu in education_list:
                degree = edu.get('degree', {})
                institution = edu.get('institution', {})
                dates = edu.get('dates', {})
                achievements = edu.get('achievements', {})
                
                education = {
                    'candidate_id': i,
                    'degree_level': degree.get('level', 'Unknown'),
                    'field': degree.get('field', 'Unknown'),
                    'institution': institution.get('name', 'Unknown'),
                    'institution_location': institution.get('location', 'Unknown'),
                    'graduation_date': dates.get('expected_graduation', 'Unknown'),
                    'gpa': achievements.get('gpa', None)
                }
                educations.append(education)
        dataframes['educations'] = pd.DataFrame(educations)
        
        # Skills data
        skills = []
        for i, record in enumerate(self.data):
            skills_data = record.get('skills', {})
            technical = skills_data.get('technical', {})
            
            # Programming languages
            prog_langs = technical.get('programming_languages', [])
            for lang in prog_langs:
                if isinstance(lang, dict) and lang.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'programming_language',
                        'skill_name': lang.get('name', 'Unknown'),
                        'skill_level': lang.get('level', 'Unknown')
                    })
            
            # Frameworks
            frameworks = technical.get('frameworks', [])
            for fw in frameworks:
                if isinstance(fw, dict) and fw.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'framework',
                        'skill_name': fw.get('name', 'Unknown'),
                        'skill_level': fw.get('level', 'Unknown')
                    })
            
            # Databases
            databases = technical.get('databases', [])
            for db in databases:
                if isinstance(db, dict) and db.get('name', 'Unknown') != 'Unknown':
                    skills.append({
                        'candidate_id': i,
                        'skill_type': 'database',
                        'skill_name': db.get('name', 'Unknown'),
                        'skill_level': db.get('level', 'Unknown')
                    })
        
        dataframes['skills'] = pd.DataFrame(skills)
        
        return dataframes
    
    def data_overview(self):
        """Print comprehensive data overview"""
        print("=== RESUME DATA OVERVIEW ===\n")
        print(f"Total number of resumes: {len(self.data)}")
        
        for table_name, df in self.df.items():
            print(f"\n{table_name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            if not df.empty:
                print(f"  Sample data:")
                print(df.head(2).to_string(index=False))
        
        # Data quality assessment
        print("\n=== DATA QUALITY ASSESSMENT ===")
        for table_name, df in self.df.items():
            print(f"\n{table_name.upper()}:")
            unknown_counts = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    unknown_count = df[col].str.contains('Unknown', na=False).sum()
                    null_count = df[col].isnull().sum()
                    unknown_counts[col] = {'unknown': unknown_count, 'null': null_count}
            
            for col, counts in unknown_counts.items():
                if counts['unknown'] > 0 or counts['null'] > 0:
                    print(f"  {col}: {counts['unknown']} 'Unknown', {counts['null']} null")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Resume Data Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Geographic distribution
        location_counts = self.df['candidates']['city'].value_counts()
        location_counts = location_counts[location_counts.index != 'Unknown']
        if not location_counts.empty:
            axes[0,0].pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%')
            axes[0,0].set_title('Geographic Distribution of Candidates')
        
        # 2. Experience levels
        if not self.df['experiences'].empty:
            level_counts = self.df['experiences']['level'].value_counts()
            axes[0,1].bar(level_counts.index, level_counts.values)
            axes[0,1].set_title('Experience Levels')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Education levels
        if not self.df['educations'].empty:
            edu_counts = self.df['educations']['degree_level'].value_counts()
            axes[0,2].bar(edu_counts.index, edu_counts.values)
            axes[0,2].set_title('Education Levels')
        
        # 4. Top Programming Languages
        if not self.df['skills'].empty:
            prog_langs = self.df['skills'][self.df['skills']['skill_type'] == 'programming_language']
            if not prog_langs.empty:
                lang_counts = prog_langs['skill_name'].value_counts().head(10)
                axes[1,0].barh(lang_counts.index, lang_counts.values)
                axes[1,0].set_title('Top Programming Languages')
        
        # 5. Skill levels distribution
        if not self.df['skills'].empty:
            skill_level_counts = self.df['skills']['skill_level'].value_counts()
            axes[1,1].pie(skill_level_counts.values, labels=skill_level_counts.index, autopct='%1.1f%%')
            axes[1,1].set_title('Skill Level Distribution')
        
        # 6. Employment types
        if not self.df['experiences'].empty:
            emp_type_counts = self.df['experiences']['employment_type'].value_counts()
            axes[1,2].bar(emp_type_counts.index, emp_type_counts.values)
            axes[1,2].set_title('Employment Types')
            axes[1,2].tick_params(axis='x', rotation=45)
        
        # 7. Technologies word cloud data preparation
        all_technologies = []
        for record in self.data:
            experience_list = record.get('experience', [])
            for exp in experience_list:
                tech_env = exp.get('technical_environment', {})
                technologies = tech_env.get('technologies', [])
                all_technologies.extend(technologies)
        
        tech_counter = Counter([tech for tech in all_technologies if tech != 'Unknown'])
        if tech_counter:
            # Create a simple bar chart instead of word cloud for compatibility
            top_techs = dict(tech_counter.most_common(10))
            axes[2,0].barh(list(top_techs.keys()), list(top_techs.values()))
            axes[2,0].set_title('Top Technologies')
        
        # 8. Companies mentioned
        if not self.df['experiences'].empty:
            company_counts = self.df['experiences']['company'].value_counts().head(10)
            company_counts = company_counts[company_counts.index != 'Unknown']
            if not company_counts.empty:
                axes[2,1].barh(company_counts.index, company_counts.values)
                axes[2,1].set_title('Top Companies')
        
        # 9. Data completeness heatmap
        completeness_data = {}
        for table_name, df in self.df.items():
            completeness = {}
            for col in df.columns:
                if col != 'candidate_id':
                    if df[col].dtype == 'object':
                        complete_ratio = 1 - (df[col].str.contains('Unknown', na=False).sum() + df[col].isnull().sum()) / len(df)
                    else:
                        complete_ratio = 1 - df[col].isnull().sum() / len(df)
                    completeness[col] = complete_ratio
            completeness_data[table_name] = completeness
        
        # Create a simple completeness visualization
        axes[2,2].text(0.1, 0.9, 'Data Completeness:', fontsize=12, fontweight='bold', transform=axes[2,2].transAxes)
        y_pos = 0.8
        for table, cols in completeness_data.items():
            axes[2,2].text(0.1, y_pos, f"{table}:", fontsize=10, fontweight='bold', transform=axes[2,2].transAxes)
            y_pos -= 0.1
            for col, ratio in cols.items():
                axes[2,2].text(0.15, y_pos, f"{col}: {ratio:.1%}", fontsize=8, transform=axes[2,2].transAxes)
                y_pos -= 0.05
                if y_pos < 0.1:
                    break
            y_pos -= 0.05
        axes[2,2].set_xlim(0, 1)
        axes[2,2].set_ylim(0, 1)
        axes[2,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Skills Distribution', 'Geographic Distribution', 
                          'Experience Levels', 'Technology Trends'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Skills distribution
        if not self.df['skills'].empty:
            skill_counts = self.df['skills']['skill_name'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=skill_counts.values, y=skill_counts.index, orientation='h', name='Skills'),
                row=1, col=1
            )
        
        # Geographic distribution
        location_counts = self.df['candidates']['city'].value_counts()
        location_counts = location_counts[location_counts.index != 'Unknown']
        if not location_counts.empty:
            fig.add_trace(
                go.Pie(labels=location_counts.index, values=location_counts.values, name='Locations'),
                row=1, col=2
            )
        
        # Experience levels
        if not self.df['experiences'].empty:
            level_counts = self.df['experiences']['level'].value_counts()
            fig.add_trace(
                go.Bar(x=level_counts.index, y=level_counts.values, name='Experience'),
                row=2, col=1
            )
        
        # Technology trends
        all_technologies = []
        for record in self.data:
            experience_list = record.get('experience', [])
            for exp in experience_list:
                tech_env = exp.get('technical_environment', {})
                technologies = tech_env.get('technologies', [])
                all_technologies.extend(technologies)
        
        tech_counter = Counter([tech for tech in all_technologies if tech != 'Unknown'])
        if tech_counter:
            top_techs = dict(tech_counter.most_common(10))
            fig.add_trace(
                go.Bar(x=list(top_techs.keys()), y=list(top_techs.values()), name='Technologies'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Interactive Resume Data Dashboard")
        fig.show()
    
    def export_summary_report(self, output_file='resume_data_summary.txt'):
        """Export a comprehensive text summary"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RESUME DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Resumes Analyzed: {len(self.data)}\n\n")
            
            # Geographic analysis
            f.write("GEOGRAPHIC DISTRIBUTION:\n")
            location_counts = self.df['candidates']['city'].value_counts()
            for city, count in location_counts.items():
                f.write(f"  {city}: {count} candidates\n")
            f.write("\n")
            
            # Skills analysis
            f.write("SKILLS ANALYSIS:\n")
            if not self.df['skills'].empty:
                skill_counts = self.df['skills']['skill_name'].value_counts()
                f.write("Top Skills:\n")
                for skill, count in skill_counts.head(15).items():
                    f.write(f"  {skill}: {count} candidates\n")
            f.write("\n")
            
            # Experience analysis
            f.write("EXPERIENCE ANALYSIS:\n")
            if not self.df['experiences'].empty:
                level_counts = self.df['experiences']['level'].value_counts()
                for level, count in level_counts.items():
                    f.write(f"  {level}: {count} positions\n")
            f.write("\n")
            
            # Data quality
            f.write("DATA QUALITY SUMMARY:\n")
            for table_name, df in self.df.items():
                unknown_percentage = 0
                total_cells = 0
                for col in df.columns:
                    if df[col].dtype == 'object' and col != 'candidate_id':
                        unknown_count = df[col].str.contains('Unknown', na=False).sum()
                        null_count = df[col].isnull().sum()
                        total_cells += len(df)
                        unknown_percentage += unknown_count + null_count
                
                if total_cells > 0:
                    completion_rate = 1 - (unknown_percentage / total_cells)
                    f.write(f"  {table_name}: {completion_rate:.1%} data completeness\n")
        
        print(f"Summary report exported to: {output_file}")

# Usage example
def analyze_resume_data(jsonl_file_path):
    """Main function to analyze resume data"""
    visualizer = ResumeDataVisualizer(jsonl_file_path)
    
    # Print overview
    visualizer.data_overview()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer.create_visualizations()
    
    # Create interactive dashboard (if plotly is available)
    try:
        print("Creating interactive dashboard...")
        visualizer.create_interactive_dashboard()
    except ImportError:
        print("Plotly not available. Skipping interactive dashboard.")
    
    # Export summary
    visualizer.export_summary_report()
    
    return visualizer

