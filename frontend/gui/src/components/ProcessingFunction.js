import React from 'react';
import classname from 'classnames'
import {Header,Menu} from 'semantic-ui-react';

// eslint-disable-next-line
const functions = [
    {
        name :'Załaduj obraz',
        function : 'load_image'
    },
    {
        name :'Grayscale',
        function : 'grayscale_luma'
    },
    {
        name :'Binaryzacja',
        function : 'otsu'
    },
    {
        name :'Erozja',
        function : 'erosion'
    },
];

const displayListFunction = (functions, activeItem, handler,disabled) => functions.map((el) => (

    <Menu.Item
        key={el.name}
        name={el.name}
        active={activeItem === el.name}
        onClick={handler}
        disabled={disabled}
    >
        <Header as='h5'>{el.name}</Header>
    </Menu.Item>
));

const ProcessingFunctions = (props) => {
    const menuClass = classname('menuCl');
    return (
        <Menu vertical compact className={menuClass}>
            {displayListFunction(props.functions,props.activeItem, props.onClick, props.disabled)}
        </Menu>
    )
}

export default ProcessingFunctions;