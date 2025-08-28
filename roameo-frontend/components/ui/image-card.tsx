import React from 'react';

interface ImageCardProps {
  children?: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

const styles = {
  Card: {
    top: '101px',
    left: '56px',
    width: '1312px',
    height: '349px',
    backgroundColor: 'rgba(3,3,3,0.48)',
    borderRadius: '32px',
  },
};

const ImageCard: React.FC<ImageCardProps> = ({ children, className = '', style = {} }) => {
  const combinedStyle = {
    ...styles.Card,
    ...style,
  };

  return (
    <div 
      className={className}
      style={combinedStyle}
    >
      {children}
    </div>
  );
};

export default ImageCard;