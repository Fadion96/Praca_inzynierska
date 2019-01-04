import React from 'react';
import classname from 'classnames'
import {Header, Label, Menu} from 'semantic-ui-react';


const displayListFunction = (functions, activeItem, handler, disabled, key) => functions.map((el) => {
        if (el.type === key) {
            return <Menu.Item
                key={el.name}
                name={el.name}
                active={activeItem === el.name}
                onClick={handler}
                disabled={disabled}
            >
                <Header as='h5'>{el.name}</Header>
            </Menu.Item>
        }
    }
);

const ProcessingFunctions = (props) => {
    const menuClass = classname('functions-menu');
    const menuLabel = classname('menu-label');
    return (
        <Menu vertical fluid className={menuClass}>
            <Menu.Item as={Label} color={'black'} className={menuLabel}>
                <Header as='h5' inverted>Operacje na dowolnym obrazie</Header>
            </Menu.Item>
            {displayListFunction(props.functions, props.activeItem, props.onClick, props.disabled, "all")}
            <Menu.Item as={Label} color={'black'} className={menuLabel}>
                <Header as='h5' inverted>Operacje na kolorowym obrazie</Header>
            </Menu.Item>
            {displayListFunction(props.functions, props.activeItem, props.onClick, props.disabled, "color")}
            <Menu.Item as={Label} color={'black'} className={menuLabel}>
                <Header as='h5' inverted>Operacje na obrazie w skali szarości</Header>
            </Menu.Item>
            {displayListFunction(props.functions, props.activeItem, props.onClick, props.disabled, "grayscale")}
            <Menu.Item as={Label} color={'black'} className={menuLabel}>
                <Header as='h5' inverted>Operacje na obrazie binarnym </Header>
            </Menu.Item>
            {displayListFunction(props.functions, props.activeItem, props.onClick, props.disabled, "binary")}
            <Menu.Item as={Label} color={'black'} className={menuLabel}>
                <Header as='h5' inverted>Operacje użytkownika</Header>
            </Menu.Item>
            {displayListFunction(props.functions, props.activeItem, props.onClick, props.disabled, "user")}
        </Menu>
    )
};

export default ProcessingFunctions;